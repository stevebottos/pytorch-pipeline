# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from SSD.SSD_sgrvinod.utils import *
import numpy as np


class SSD(nn.Module):
    """
    This is an adaptation of Nvidia's SSD implementation, with additions from
    sgrvinod's implementation
    """
    def __init__(self, backbone, num_classes):
        super().__init__()

        self.feature_extractor = backbone

        self.label_num = num_classes  # number of COCO classes
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []
        self.priors_cxcy = self.create_prior_boxes()

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.label_num, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()

        return locs, confs

    def forward(self, x):
        x = self.feature_extractor(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        locs = locs.transpose(1,2) # Added becsuse the output shape is actually not what's above...
        confs = confs.transpose(1,2) # Same reason...
        return locs, confs


    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
            """
            TAKEN FROM https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py

            Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
            For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
            :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
            :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
            :param min_score: minimum threshold for a box to be considered a match for a certain class
            :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
            :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
            :return: detections (boxes, labels, and scores), lists of length batch_size
            """
            batch_size = predicted_locs.size(0)
            n_priors = self.priors_cxcy.size(0)
            predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

            # Lists to store final predicted boxes, labels, and scores for all images
            all_images_boxes = list()
            all_images_labels = list()
            all_images_scores = list()

            assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

            for i in range(batch_size):
                # Decode object coordinates from the form we regressed predicted boxes to
                decoded_locs = cxcy_to_xy(
                    gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

                # Lists to store boxes and scores for this image
                image_boxes = list()
                image_labels = list()
                image_scores = list()

                max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

                # Check for each class
                for c in range(1, self.label_num):
                    # Keep only predicted boxes and scores where scores for this class are above the minimum score
                    class_scores = predicted_scores[i][:, c]  # (8732)
                    score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                    n_above_min_score = score_above_min_score.sum().item()
                    if n_above_min_score == 0:
                        continue

                    class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                    class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                    # Sort predicted boxes and scores by scores
                    class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                    class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                    # Find the overlap between predicted boxes
                    overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                    # Non-Maximum Suppression (NMS)

                    # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                    # 1 implies suppress, 0 implies don't suppress
                    suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                    # Consider each box in order of decreasing scores
                    for box in range(class_decoded_locs.size(0)):
                        # If this box is already marked for suppression
                        if suppress[box] == 1:
                            continue

                        # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                        # Find such boxes and update suppress indices
                        condition = overlap[box] > max_overlap
                        condition = torch.tensor(condition, dtype=torch.uint8).to(device)
                        suppress = torch.max(suppress, condition)
                        # The max operation retains previously suppressed boxes, like an 'OR' operation

                        # Don't suppress this box, even though it has an overlap of 1 with itself
                        suppress[box] = 0

                    # Store only unsuppressed boxes for this class
                    image_boxes.append(class_decoded_locs[1 - suppress])
                    image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                    image_scores.append(class_scores[1 - suppress])

                # If no object in any class is found, store a placeholder for 'background'
                if len(image_boxes) == 0:
                    image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                    image_labels.append(torch.LongTensor([0]).to(device))
                    image_scores.append(torch.FloatTensor([0.]).to(device))

                # Concatenate into single tensors
                image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
                image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
                image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
                n_objects = image_scores.size(0)

                # Keep only the top k objects
                if n_objects > top_k:
                    image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                    image_scores = image_scores[:top_k]  # (top_k)
                    image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                    image_labels = image_labels[sort_ind][:top_k]  # (top_k)

                # Append to lists that store predicted boxes and scores for all images
                all_images_boxes.append(image_boxes)
                all_images_labels.append(image_labels)
                all_images_scores.append(image_scores)

            return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

    def create_prior_boxes(self):
        """
        TAKEN FROM https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py

        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * np.sqrt(ratio), obj_scales[fmap] / np.sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = np.sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0/dboxes.scale_xy
        self.scale_wh = 1.0/dboxes.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = nn.CrossEntropyLoss(reduce=False)

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.dboxes[:, :2, :])/self.dboxes[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._loc_vec(gloc)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float()*sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        #print(con.shape, mask.shape, neg_mask.shape)
        closs = (con*(mask.float() + neg_mask.float())).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss*num_mask/pos_num).mean(dim=0)
        return ret

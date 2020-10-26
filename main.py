"""
Pre-trained options:

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)

The models subpackage (models.detection) contains definitions for the following model architectures for detection:

Faster R-CNN ResNet-50 FPN
Mask R-CNN ResNet-50 FPN

* Note that some options, like mobilenetv1, are missing. These can be added
easily though.

* We'll do mobilenetv2 for this runthrough
"""
import os

import torch
import torchvision
from torchvision import models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import references.detection.utils as utils
from references.detection.engine import train_one_epoch, evaluate

from helper import pascal_encode_labels, ObjectDetectionDataset, get_transform

# This can change because the images and annotations may be in some other directory
ROOT = "/home/steve/repos/pytorch-pipeline"
IMAGES = os.path.join(ROOT, "Sparrowatch/JPEGImages")
ANNOS = os.path.join(ROOT, "Sparrowatch/Annotations")

# Get some relevant info from the dataset
label_encodings = pascal_encode_labels(ANNOS)
num_classes = len(label_encodings) + 1 # Add 1 for background

def main():
    """
    Mobilenet backbone
    With mobilenet, we can remove the FC layers easily by grabbing only the
    "features" layers. Recall that we want to remove the FC layers so that
    we can feed into the SSD for object detection rather than image classification
    """
    fullmodel = torchvision.models.mobilenet_v2(pretrained=True)
    backbone = fullmodel.features
    backbone.out_channels = 1280
    """
    Resnet backbone
    With Resnet, we can't just grab "features" so we have to remove the FC
    layer manually
    """
    # fullmodel = torchvision.models.resnet50(pretrained=True)
    # backbone = torch.nn.Sequential(*(list(fullmodel.children())[:-1]))
    # backbone.out_channels = 2048

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)


    # use our dataset and defined transformations
    dataset = ObjectDetectionDataset(
                    imdir=IMAGES,
                    annodir=ANNOS,
                    label_encodings=label_encodings,
                    format="pascal",
                    transforms=get_transform(train=True))

    dataset_test = ObjectDetectionDataset(
                    imdir=IMAGES,
                    annodir=ANNOS,
                    label_encodings=label_encodings,
                    format="pascal",
                    transforms=get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    train_val_split_point = int(len(indices) * 0.7)
    dataset = torch.utils.data.Subset(dataset, indices[:train_val_split_point])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[train_val_split_point:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # Move the model to the right device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model.to(device)

    # Construct the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)


if __name__ == "__main__":
    main()

import os
from PIL import Image
import xml.etree.ElementTree as ET
import torch
import collections

import references.detection.utils as utils
import references.detection.transforms as T


def make_dataset(images_path,
                annotations_path,
                train_batch_size,
                val_batch_size,
                train_fraction,
                format="pascal"):

    if format == "pascal":
        label_encodings = pascal_encode_labels(annotations_path)

    num_classes = len(label_encodings) + 1 # Add 1 for background

    dataset = ObjectDetectionDataset(
                    imdir=images_path,
                    annodir=annotations_path,
                    label_encodings=label_encodings,
                    format=format,
                    transforms=get_transform(train=True))

    dataset_test = ObjectDetectionDataset(
                    imdir=images_path,
                    annodir=annotations_path,
                    label_encodings=label_encodings,
                    format=format,
                    transforms=get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    train_val_split_point = int(len(indices) * train_fraction)
    dataset = torch.utils.data.Subset(dataset, indices[:train_val_split_point])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[train_val_split_point:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=val_batch_size, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    return data_loader, data_loader_test, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # Augmentations go here... Won't do any for now
        pass
        # transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def pascal_encode_labels(annodir):

    labels = []
    for anno in os.listdir(annodir):
        anno_path = os.path.join(annodir, anno)
        tree = ET.parse(anno_path)
        objects = tree.findall("object")
        for obj in objects:
            labels.append(obj.find("name").text)

    labels = sorted(list(set(labels))) # To keep order consistency

    label_encodings = {}
    for i, l in enumerate(labels):
        label_encodings[l] = i+1

    return label_encodings


class ObjectDetectionDataset(object):
    def __init__(self,
                imdir,
                annodir,
                label_encodings=None,
                format="pascal",
                transforms=None):
        self.imdir = imdir
        self.annodir = annodir
        self.format = format
        self.transforms = transforms
        self.label_encodings = label_encodings

        self.imgs = list(sorted(os.listdir(imdir)))
        self.annos = list(sorted(os.listdir(annodir)))

    def pascal_extract_bbox_coordinates(self, idx):
        img_path = os.path.join(self.imdir, self.imgs[idx])
        anno_path = os.path.join(self.annodir, self.annos[idx])
        img = Image.open(img_path).convert("RGB")

        tree = ET.parse(anno_path)
        objects = tree.findall("object")

        boxes = []
        labels = []
        areas = []
        crowds = []
        for obj in objects:
            label = obj.find("name").text
            bbs = obj.find("bndbox")
            xmin = float(bbs[0].text)
            xmax = float(bbs[2].text)
            ymin = float(bbs[1].text)
            ymax = float(bbs[3].text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.label_encodings[label])
            areas.append(xmax-xmin * ymax-ymin)
            crowds.append(0)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(crowds, dtype=torch.uint8)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __getitem__(self, idx):
        if self.format == "pascal":
            img, target = self.pascal_extract_bbox_coordinates(idx)
        return img, target

    def __len__(self):
        return len(self.imgs)

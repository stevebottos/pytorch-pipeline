import os

import torch
import torchvision
from torchvision import models
from torchvision.models.detection import FasterRCNN


import dataset as ds
import basemodel as bm
import framework as fm
import pipeline as pp

# This can change because the images and annotations may be in some other directory
ROOT = "/home/steve/repos/pytorch"
IMAGES = os.path.join(ROOT, "Sparrowatch/JPEGImages")
ANNOS = os.path.join(ROOT, "Sparrowatch/Annotations")

def train(config):

    basemodel_name = config["basemodel"]
    framework = config["framework"]
    lr = config["lr"]
    momentum = config["momentum"]
    weight_decay = config["weight_decay"]
    train_batch_size = config["train_bs"]
    val_batch_size = config["val_bs"]

    data_loader, data_loader_test, num_classes = ds.make_dataset(IMAGES,
                                                            ANNOS,
                                                            train_batch_size=train_batch_size,
                                                            val_batch_size=val_batch_size,
                                                            train_fraction=0.7)

    basemodel = bm.fetch_basemodel(basemodel_name, framework)
    model = fm.set_basemodel_in_framework(basemodel, framework, num_classes)

    # some pipeline configs (put these into the pipeline module )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if framework == "FasterRCNN":
        pipeline = pp.PipelineFasterRCNN(num_epochs=10,
                                         model=model,
                                         lr=lr, momentum=momentum, weight_decay=weight_decay,
                                         data_loader=data_loader,
                                         data_loader_test=data_loader_test,
                                         device=device,
                                         print_freq=1)

    elif framework == "SSD":
        pipeline = pp.PipelineSSD(num_epochs=10,
                                         model=model,
                                         lr=lr, momentum=momentum, weight_decay=weight_decay,
                                         data_loader=data_loader,
                                         data_loader_test=data_loader_test,
                                         device=device,
                                         print_freq=1)
    pipeline.train()

if __name__ == "__main__":
    # This will be passed in through the CLI/NB/desktop
    config = {
    "framework" : "SSD",
    "basemodel" : "resnet50",
    "lr" : 0.005,
    "momentum" : 0.9,
    "weight_decay" : 0.0005,
    "train_bs" : 2,
    "val_bs" : 2
    }
    train(config)

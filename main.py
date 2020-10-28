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
ROOT = "/home/steve/repos/pytorch-pipeline"
IMAGES = os.path.join(ROOT, "Sparrowatch/JPEGImages")
ANNOS = os.path.join(ROOT, "Sparrowatch/Annotations")

def train(config):

    basemodel_name = config["basemodel"]
    framework = config["framework"]

    data_loader, data_loader_test, num_classes = ds.make_dataset(IMAGES, ANNOS,
                                                        train_fraction=0.7)
    basemodel = bm.fetch_basemodel(basemodel_name, framework)
    model = fm.set_basemodel_in_framework(basemodel, framework, num_classes)

    # some pipeline configs (put these into the pipeline module )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    pipeline = pp.PipelineFasterRCNN(num_epochs=10,
                                     model=model,
                                     optimizer=optimizer,
                                     data_loader=data_loader,
                                     data_loader_test=data_loader_test,
                                     device=device,
                                     print_freq=1)
    pipeline.train()

if __name__ == "__main__":
    # This will be passed in through the CLI/NB/desktop
    config = {
    "framework" : "SSD300",
    "basemodel" : "resnet50"
    }
    train(config)

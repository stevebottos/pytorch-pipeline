import torch
import torch.nn as nn
import torchvision
from torchvision import models

def fetch_basemodel(basemodel, framework):
    available_models = {
                        "mobilenet_v2" : mobilenet_v2,
                        "resnet50" : resnet50,
                        }
    if basemodel not in available_models:
        models = "".join(["\t"+str(am)+"\n" for am in available_models])
        exception_message = "\nPlease choose a model from:\n" + models
        raise Exception(exception_message)

    return available_models[basemodel](framework)


"""
Models can be used with the FasterRCNN framework out of the box with pytorch

SSD backbones need to fit Nvidia's SSD framework implementation,
which changes some a few things (for differences check the ResNet class
definition in the Pytorch source code)
"""
def mobilenet_v2(framework):
    if framework == 'FasterRCNN':
        fullmodel = torchvision.models.mobilenet_v2(pretrained=True)
        backbone = fullmodel.features
        backbone.out_channels = 1280

    return backbone


def resnet50(framework):
    if framework == "FasterRCNN":
        fullmodel = models.resnet50(pretrained=True)
        backbone = torch.nn.Sequential(*(list(fullmodel.children())[:-1]))
        backbone.out_channels = 2048

    elif framework == "SSD300":
        backbone = ResNetSSD(backbone='resnet50')

    return backbone

class ResNetSSD(nn.Module):
    def __init__(self, backbone='resnet50', backbone_path=None):
        super().__init__()
        if backbone == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = models.resnet101(pretrained=True)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  
            backbone = models.resnet152(pretrained=True)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))


        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

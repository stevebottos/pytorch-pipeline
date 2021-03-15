from torchvision.models.detection import FasterRCNN
from SSD.SSD_nvidia.src.model import SSD

def set_basemodel_in_framework(basemodel,
                               framework,
                               num_classes,
                               rpn_anchor_generator=None,
                               box_roi_pooler=None
                               ):

        """
        FasterRCNN frameworks require rpn_anchor or box_roi_pool to be
        specified. If they're not specified PyTorch will just use defaults which
        are fine.

        If a custom rpn_anchor_generator/roi_pooler is desired, define them as
        follows (example)...
          anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
                                             sizes=((32, 64, 128, 256, 512),),
                                             aspect_ratios=((0.5, 1.0, 2.0),))

          roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                          output_size=7,
                                                          sampling_ratio=2)
        ... and pass them into the function

        SSD models don't need rpn anchor generators or roi poolers
        """

        if framework == "FasterRCNN":
            model = FasterRCNN(basemodel,
                               num_classes=num_classes,
                               rpn_anchor_generator=None,
                               box_roi_pool=None)

        elif framework == "SSD":
            model = SSD(basemodel, num_classes=num_classes)

        return model

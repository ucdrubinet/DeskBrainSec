import torch.nn as nn
import torchvision
import torch

class PlaqueTissueClassifier(nn.Module):
    def __init__(self):
        super(PlaqueTissueClassifier, self).__init__()
        backbone = torchvision.models.mobilenet_v3_small(torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        # plaque classes: diffuse, cored, CAA
        self.plaque_classifier = nn.Linear(backbone.classifier[0].in_features, 3)
        # tissue classes: white matter, gray matter, background
        self.tissue_classifier = nn.Linear(backbone.classifier[0].in_features, 3)

    # returns plaque and tissue predictions
    def forward(self, x):
        feats = self.features(x)
        feats = torch.flatten(self.avgpool(feats), 1)
        plaque = self.plaque_classifier(feats)
        tissue = self.tissue_classifier(feats)
        return plaque, tissue

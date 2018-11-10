import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.PreTrainedVGG = models.vgg19_bn(pretrained=True)
        
        ct = 0
        for name, child in self.PreTrainedVGG.named_children():
            ct += 1
            if ct < 5:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(64, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, nclasses),
        )
        self.PreTrainedVGG.classifier = self.classifier

    def forward(self, x):
        return self.PreTrainedVGG.forward(x)

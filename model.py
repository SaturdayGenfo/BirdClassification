import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.PreTrainedResNet = models.resnet18(pretrained=True)
        
        
        ct = 0
        for name, child in self.PreTrainedResNet.named_children():
            ct += 1
            if ct < 3:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
        
        self.PreTrainedResNet.avgpool = nn.AvgPool2d(2, stride=1)
        self.numfeatures = self.PreTrainedResNet.fc.in_features
        self.PreTrainedResNet.fc = nn.Sequential(
            nn.Linear(4068, 4096),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, nclasses),
        )

    def forward(self, x):
        return self.PreTrainedResNet.forward(x)

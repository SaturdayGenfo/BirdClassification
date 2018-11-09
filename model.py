import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.PreTrainedResNet = models.resnet18(pretrained=True)
        for param in self.PreTrainedResNet.parameters():
            param.requires_grad = False
        self.numfeatures = self.PreTrainedResNet.fc.in_features
        self.PreTrainedResNet.fc = nn.Linear(self.numfeatures, 50)

    def forward(self, x):
        return self.PreTrainedResNet.forward(x)

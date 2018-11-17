import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.vgg16(pretrained=False, num_classes=nclasses)

    def forward(self, x):
        return self.model.forward(x)

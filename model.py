import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, nclasses)
        
        '''
        ct = 0
        for name, child in self.PreTrainedResNet.named_children():
            ct += 1
            if ct < 5:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
        '''

    def forward(self, x):
        return self.model.forward(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.PreTrainedResNet = models.alexnet(pretrained=True)
        
        '''
        ct = 0
        for name, child in self.PreTrainedResNet.named_children():
            ct += 1
            if ct < 5:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
        '''
        
        self.PreTrainedResNet.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, nclasses),
        )

    def forward(self, x):
        return self.PreTrainedResNet.forward(x)

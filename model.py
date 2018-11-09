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
            if ct < 2:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
        self.num_features = self.PreTrainedVGG.classifier[6].in_features
        self.features = list(self.PreTrainedVGG.classifier.children())[:-1] # Remove last layer
        self.features.extend([nn.Linear(num_features, nclasses)]) # Add our layer with 4 outputs
        self.PreTrainedVGG.classifier = nn.Sequential(*features) 

    def forward(self, x):
        return self.PreTrainedVGG.forward(x)

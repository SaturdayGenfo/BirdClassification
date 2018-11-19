import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.model = models.squeezenet(pretrained=True)
        self.model = models.squeezenet1_0(pretrained=True)
        self.model_ft.num_classes = nclasses
        self.model.classifier[1] = nn.Conv2d(512, nclasses, kernel_size=(1,1), stride=(1,1))
        '''
        ct = 0
        for name, child in self.model.named_children():
            ct += 1
            if ct < 5:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
        '''
        

    def forward(self, x):
        return self.model.forward(x)

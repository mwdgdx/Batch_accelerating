
from __future__ import print_function
import torch.nn as nn
import torch .nn. functional as F
from math import sqrt

class c1_model(nn.Module):
    def __init__(self,num_classes=10,init_weights=True):
        super(c1_model,self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=5,stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 5, stride = 1, padding = 2)
        self.bn2=nn.BatchNorm2d(64)
        self.fc1=nn.Linear(64*8*8, 384)
        self.fc2=nn.Linear(384,192)
        self.fc3 = nn.Linear(192, 10)
        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x=F.max_pool2d(self.bn1(F.relu(self.conv1(x))),3,2,1)
        x=F.max_pool2d(self.bn2(F.relu(self.conv2(x))),3,2,1)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for fc in [self.fc1, self.fc2, self.fc3]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 0.1/ sqrt(C_in))
            nn.init.constant_(fc.bias, 0.01)
        for conv in [self.conv1, self.conv2]:
            nn.init.normal_(conv.weight, 0.0, 0.01)
            nn.init.constant_(conv.bias, 0)
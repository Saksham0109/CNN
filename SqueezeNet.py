import torch.nn as nn
import torch
class SqueezeNet(nn.Module):
    def Fire(self,in_channels,out_channels):
        squeeze=nn.Conv2d(in_channels=in_channels,out_channels=out_channels//8,kernel_size=1)
        expand1=nn.Conv2d(in_channels=out_channels//8,out_channels=out_channels//2,kernel_size=1)
        expand3=nn.Conv2d(in_channels=out_channels//2,out_channels=out_channels,kernel_size=3,padding=1)
        relu=nn.ReLU()
        return nn.Sequential(squeeze,relu,expand1,relu,expand3,relu)

    def FireGroup(self,channels):
        layers=[]
        for i in range(1,len(channels)):
            layers.append(self.Fire(channels[i-1],channels[i]))
        return nn.Sequential(*layers)
    
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
        self.fire1=self.FireGroup([96,128,128,256])
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
        self.fire2=self.FireGroup([256,256,384,384,512])
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
        self.fire3=self.FireGroup([512,512])
        self.conv2=nn.Conv2d(in_channels=512,out_channels=1000,kernel_size=1)
        self.avgpool=nn.AvgPool2d(kernel_size=13)
        self.relu=nn.ReLU()
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire1(x)
        x=self.maxpool2(x)
        x = self.fire2(x)
        x = self.maxpool3(x)
        x = self.fire3(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1000)
        return x

    

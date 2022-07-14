import nntplib
from torch import nn 
VGG19=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self,in_channel=3,num_classes=100):
        super(VGG,self).__init__()
        self.in_channel=in_channel
        self.NN=self.conv_layers(VGG19,in_channel)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
            )
        self.flatten=nn.Flatten(start_dim=1)
    
    def forward(self,x):
        x=self.NN(x)
        x=self.flatten(x)
        x=self.classifier(x)
        return x


    def conv_layers(self, architecture,in_channel):
        layers = []
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channel,out_channels=out_channels,
                                     kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channel = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
                
        return nn.Sequential(*layers)
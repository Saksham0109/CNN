from torch import nn
class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.NIN=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=3,kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=3,kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=3,kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
        self.flatten=nn.Flatten()
        self.l=nn.Linear(48,10)
        
    def forward(self,x):
        x=self.NIN(x)
        x=self.flatten(x)
        x=self.l(x)
        return x
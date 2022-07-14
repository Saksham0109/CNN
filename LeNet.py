from torch import nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2)
        self.s2=nn.AvgPool2d(kernel_size=2,stride=2)
        self.c3=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0)
        self.s4=nn.AvgPool2d(kernel_size=2,stride=2)
        self.c5=nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1)
        self.flatten= nn.Flatten(start_dim=1)
        self.l1=nn.Linear(in_features=120,out_features=84)
        self.l2=nn.Linear(in_features=84,out_features=10)
        self.act=nn.Tanh()
    def forward(self,x):
        x=self.c1(x)
        x=self.act(x)
        x=self.s2(x)
        x=self.c3(x)
        x=self.act(x)
        x=self.s4(x)
        x=self.c5(x)
        x=self.act(x)
        x=self.flatten(x)
        x=self.l1(x)
        x=self.act(x)
        x=self.l2(x)
        return x
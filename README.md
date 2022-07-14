# CNN
This repository is a compilation of all CNN architectures i have written alongside with a training jupyter notebook which can be used to train them on FashionMNIST or CIFAR10 dataset.

The notebook gives a summary of the model,tell the number of FLOPS for a single image and then trains and saves the model according to the training schedule that is set.

It also takes tensorboard log of the accuracy and loss function.

NOTE:Since the two datasets have different input size therefore any model will not work on both of them.Am still working on that.For now GoogleNet and NIN will work on CIFAR10 and Lenet will work on FshionMNIST.AlexNet and VGG were initially made for CIFAR100. 

import os
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


cifar_labels = ('plane', 'car', 'bird', 'cat','deer', 'dog',
                 'frog', 'horse', 'ship', 'truck')

current_dir = os.getcwd()

if os.path.exists("cifar-10-python.tar.gz"):
    trainset = torchvision.datasets.CIFAR10(root = current_dir,
                                            train=False,
                                            download=False,
                                            transform=transforms.ToTensor())
else:
    trainset = torchvision.datasets.CIFAR10(root = current_dir,
                                            train=False,
                                            download=True,
                                            transform=transforms.ToTensor())


def show_model_structure():
    vgg19 = models.vgg19(num_classes=10)
    input_channel_dim = (3, 32, 32)
    summary(vgg19)


def show_images():
    batch_size = 9
    train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2)
    dataiter = iter(train_loader)
    imgs, labels = next(dataiter)

    counter = 0
    for img, label_id in zip(imgs, labels):
        img = np.transpose(img, (1, 2, 0)) # rearrange image dims from (3x32x32) to (32x32x3)

        # plot images in 3x3 grid
        plt.subplot(331 + counter).set_title(f'{cifar_labels[label_id]}')
        plt.imshow(img)
        counter += 1

    # making plots pretty DOES NOT WORK    
    plt.subplots_adjust(hspace=0.4)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]) # removes all ticks and labels
    plt.show()

def show_metrics():
    accuracy = mpimg.imread("figures\\accuracy.png")
    loss = mpimg.imread("figures\loss.png")
    plt.subplot(2,1,1)
    plt.imshow(accuracy)
    plt.subplot(2,1,2)
    plt.imshow(loss)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]) # removes all ticks and labels
    plt.show()

    


if __name__ == "__main__":
    main()
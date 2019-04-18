import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from model import c1_model
from train_common import *
import matplotlib.pyplot as plt


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
def main():
    # data loaders from cifar-10
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    tr_loader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                              shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    te_loader = torch.utils.data.DataLoader(test_set, batch_size=512,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # TODO: define model, loss function, and optimizer
    model = c1_model(False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Attempts to restore the latest checkpoint if exists
    print('Loading challenge...')
    model, start_epoch, stats = restore_checkpoint(model,'./checkpoints')
    print('training:')

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, 1000000):
        # Train model
        running_loss = 0.0
        accumulation_steps = 32
        optimizer.zero_grad()
        for i, (X, y) in enumerate(tr_loader):
            # forward + backward + optimize
            print(i)
            output = model(X)
            loss = criterion(output, y)
            loss = loss / accumulation_steps
            running_loss+=loss
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
        save_checkpoint(model, epoch+1, './checkpoints', stats)
        print(running_loss)
    print('Finished Training')
    dataiter = iter(te_loader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


if __name__ == '__main__':
    main()
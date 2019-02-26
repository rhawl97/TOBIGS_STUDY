from data_util import *

from torch.utils.data import DataLoader

trainloader = DataLoader(trainset,
                         batch_size=4,
                         shuffle=True,
                         drop_last=False)

testloader = DataLoader(testset,
                        batch_size=4,
                        shuffle=False,
                        drop_last=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

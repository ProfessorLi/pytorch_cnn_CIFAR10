"""pytorch_cnn_CIFAR10.py.

Follows the 60 minute blitz tutorial available here:
https://github.com/pytorch/tutorials/blob/master/Deep%20Learning%20with%20PyTorch.ipynb

Trains a shallow convolutional neural network to classify CIFAR10.
Current model converges after about 50 epochs, and achieves 70% test
accuracy after 100 epochs.
"""

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Hyperparameters
n_epochs = 100
batch_size = 256

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform,
)

trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size,                                          
    shuffle=True, 
    num_workers=2,
)

testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform,
)

testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=batch_size,                                      
    shuffle=False, 
    num_workers=2,
)

classes = 'plane car bird cat deer dog frog horse ship truck'.split()

# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First hidden layer
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool1 = nn.MaxPool2d(2)

        # Second hidden layer
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.pool2 = nn.MaxPool2d(2)

        # Two fully connected layers
        self.fc1   = nn.Linear(64*5*5, 120)
        self.fc2   = nn.Linear(120, 84)

        # Output layer
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        # First hidden layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Second hidden layer
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Reshape for fully connected layers
        x = x.view(-1, self.num_flat_features(x))

        # First fully connected layer
        x = self.fc1(x)
        x = F.relu(x)

        # Second fully connected layer
        x = self.fc2(x)
        x = F.relu(x)

        # Output layer
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
net = net.cuda()

# Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# Train
for epoch in range(n_epochs):  # loop over the dataset multiple times
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # Unpack minibatch
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        
        # Train on minibatch
        optimizer.zero_grad()              # initialize
        outputs = net(inputs)              # forward
        loss = criterion(outputs, labels)  # loss
        loss.backward()                    # backward
        optimizer.step()                   # update
        
        # Print accuracy
        running_loss += loss.data[0]

    print("Epoch {} / {} Train loss {:.2f}".format(
        epoch, n_epochs, running_loss))

print('Finished Training')

# Evaluate
correct = 0
total = 0
for data in testloader:
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

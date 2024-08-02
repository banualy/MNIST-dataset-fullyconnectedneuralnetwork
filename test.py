import torch
import torch.nn as nn #neural network models, loss and activation functions
import torch.optim as optim #optimization algorithms
import torch.nn.functional as F #activation functions
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Creating fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__() #initialization of method
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
input_size = 784
num_classes=10
learning_rate= 0.001
batch_size=64
num_epochs=3

#loading data
train_data = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

#initializing network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#training
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # getting data to device
        data = data.to(device=device)
        targets = targets.to(device=device)

        # correcting shape
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad() # set gradients to zero in each batch
        loss.backward()

        # gradient descent step
        optimizer.step()


# checking accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Training data")
    else:
        print("Test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions =scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(num_correct/num_samples, f'{float(num_correct)/float(num_samples)*100:.2f accuracy}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)


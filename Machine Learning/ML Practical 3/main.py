import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
])

dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [50000, 10000])
dataset_test = datasets.MNIST('./mnist', train=False, transform=transform)

batch_size = 64

train_loader = DataLoader(dataset_train, batch_size=batch_size)
valid_loader = DataLoader(dataset_valid, batch_size=batch_size)
test_loader = DataLoader(dataset_test, batch_size=batch_size)

# Display images
# fig = plt.figure()
# for i in range(16):
#     ax = fig.add_subplot(4, 4, i + 1)
#     ax.set_xticks(())
#     ax.set_yticks(())
#     ax.imshow(dataset_train[i][0].reshape(28, 28), cmap='Greys_r')

# Define network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2D, 25 filters of size 12x12x1, stride 2, no padding
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=25,
            kernel_size=12,
            stride=2,
            padding=0
        )
        # Output size = (W - K + 2P) / S + 1
        # Output: 25 x 9 x 9

        # Conv2D, 64 filters of size 5 x 5 x 25, stride 1, 
        self.conv2 = nn.Conv2d(
            in_channels=25,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )
        # Padding = 2 as 5 = (9 - 5 + 2P) / 1 + 1

        # 2x2 Max Pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer, 1024 in and 1024 out
        self.fc1 = nn.Linear(1024, 1024)
        
        # Dropout to reduce overfitting
        self.dropout = nn.Dropout(0.2)

        # Fully connected layer to predict class
        self.fc2 = nn.Linear(1024, 10)

        # Initialize weights


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1) # keeping batch dimension
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

net = Net()

# Define loss function/optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 1e-4)

# Training
for epoch in range(4):  # 2 epochs, 50000 / 64 = roughly 800 iterations per epoch, 3200 iterations total

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 iterations, validate every 100 iterations
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


            # Compute training accuracy for this batch
            _, preds = outputs.max(1)
            train_acc = (preds == labels).sum().item() / labels.size(0)
            
            # Validation accuracy
            net.eval()  # switch to eval mode
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for v_images, v_labels in valid_loader:
                    v_outputs = net(v_images)
                    v_preds = v_outputs.argmax(dim=1)
                    val_correct += (v_preds == v_labels).sum().item()
                    val_total += v_labels.size(0)
            val_acc = val_correct / val_total
            net.train()  # switch back to train mode

            print(f'Epoch {epoch+1}, Iter {i+1}, '
                  f'Loss: {running_loss/200:.3f}, '
                  f'Train Acc: {train_acc*100:.2f}%, '
                  f'Val Acc: {val_acc*100:.2f}%')
            running_loss = 0.0

print('Finished Training')

# Save model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# Test model and report accuracy
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
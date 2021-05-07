import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNet(nn.Module):
  def __init__(self, input_dim, output_dim):
    print("fcnn's output dim", output_dim)
    # Call nn.Module's constructor--don't forget this
    super().__init__()

    # Define layers
   # self.fc1 = nn.Linear(input_dim, 256)
    self.fc1 = nn.Linear(input_dim, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, output_dim)

  def forward(self, x):
    # Forward propagation
    print("first layer of fcnn", x.shape)
    x = self.fc1(x)
    x = F.relu(x)

    print("second layer of fcnn", x.shape)
    x = self.fc2(x)
    x = F.relu(x)
   

    x = self.fc3(x)

    print("fcnn's last phase,", x.shape)

    # No activation function at the end
    # nn.CrossEntropyLoss takes care of it for us

    return x

class StartingNetwork(torch.nn.Module):
  def __init__(self, input_channels, output_dim):
    print("starting net's output dim", output_dim)
    super().__init__()
    # input_channels = 3 -- red, green, blue
    self.conv1 = nn.Conv2d(input_channels, 6, 5) # 5 = filter size
    self.pool = nn.MaxPool2d(2, 2) #filter stride
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc_net = FullyConnectedNet(16*53*53, output_dim)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    print("phase 1", x.shape)
    x = self.pool(F.relu(self.conv2(x)))
    print("phase 2", x.shape)
    x = torch.reshape(x, (-1, 16 * 53 * 53))
    x = self.fc_net(x)
    print("phase 3", x.shape)

    return x

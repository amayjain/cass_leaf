import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNet(nn.Module):
  def __init__(self, input_dim, output_dim):
    #print("fcnn's output dim", output_dim)
    # Call nn.Module's constructor--don't forget this
    super().__init__()

    # Define layers
   
    self.fc1 = nn.Linear(input_dim, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, output_dim)
   
    

  def forward(self, x):
    # Forward propagation
    #print("first layer of fcnn", x.shape)
    x = self.fc1(x)
    x = F.relu(x)

    #print("second layer of fcnn", x.shape)
    x = self.fc2(x)
    x = F.relu(x)
   

    x = self.fc3(x)
    
   

    #print("fcnn's last phase,", x.shape)

    # No activation function at the end
    # nn.CrossEntropyLoss takes care of it for us

    return x

class StartingNetwork(torch.nn.Module):
  def __init__(self, input_channels, output_dim):
    print("starting net's output dim", output_dim)
    super().__init__()
    self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True) #initialize
    print("res features", self.resnet.fc.in_features)
    self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1])) #cut off last layer
    self.resnet.eval() #set on eval mode to freeze weights

    # input_channels = 3 -- red, green, blue
    #self.conv1 = nn.Conv2d(512, 6, 4, padding = (1,1)) # 5 = filter size
    #self.pool = nn.MaxPool2d(2, 2) #filter stride
    #self.conv2 = nn.Conv2d(6, 16, 4, padding = (2,2))
    #self.conv3 = nn.Conv2d(16,16,4, padding = (2,2))
    #self.fc_net = FullyConnectedNet(32*512, output_dim)

    self.fc1 = nn.Linear(2048, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 256)
    self.fc4 = nn.Linear(256, output_dim)
    #self.fc4 = nn.Linear(32, output_dim)

  def forward(self, x):
    #print("before res", x.shape)
    with torch.no_grad():
      x = self.resnet(x)
    #print("resnet shape", x.shape)
    #x = self.pool(F.relu(self.conv1(x)))
    #print("phase 1", x.shape)

    #x = self.pool(F.relu(self.conv2(x)))

    #print("phase 2", x.shape)
    #x = self.pool(F.relu(self.conv3(x)))

    #print("phase 3", x.shape)
    x = torch.reshape(x, (-1, 2048))
    #print("reshape", x.shape)
    #x = self.fc_net(x)
    #print("fc shape", x.shape)
    #print("phase 3", x.shape)
    #x = torch.argmax(x, dim = 1)
    #argmax, softmax
    x = self.fc1(x)
    #print("fc1", x.shape)
    x = F.relu(x)

    #print("second layer of fcnn", x.shape)
    x = self.fc2(x)
    x = F.relu(x)
   

    x = self.fc3(x)
    x = F.relu(x)

    x = self.fc4(x)

    #x= self.fc4(x)
    #print("fc last", x.shape)

    
    return x

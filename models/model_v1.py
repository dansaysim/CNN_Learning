import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_v1(nn.Module):

    """
    Simple CNN that has 2 Conv layers, 2 MaxPool layers, and 1 FC layer. 
    Each Conv layer goes 'conv1' -> 'relu' -> 'maxpool'
    After both layer groups, the output is flattened and then passed to a FC layer. 
    """

    def __init__(self):
        super(CNN_v1, self).__init__()

        ##Layers

        #Convoulutonal Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)

        #MaxPooling Layers
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #Dropout Layers
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)
        self.dropout_fc = nn.Dropout(p=0.5)

        #FC Layer
        self.fc1 = nn.Linear(in_features=4096, out_features=10)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize convolutional layers with Kaiming/He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Initialize fully connected layers with Xavier/Glorot initialization
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        #First conv->relu->maxpool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        #Second conv->relu->maxpool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        #Flatten
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)

        #FC Layer
        x = self.fc1(x)

        return x


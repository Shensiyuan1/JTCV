import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """
    LeNet-5 经典架构的 PyTorch 实现
    适用于 MNIST (1x28x28)
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU()
        )
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU()
        )

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        # test in mnist-> 28x28 -> 14x14 -> 7x7 -> 4x4
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1), 
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=num_classes) 
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        
        x = x.view(x.size(0), -1) 
        
        x = self.classifier(x)
        
        return x

if __name__ == '__main__':
    model = LeNet5()
    print(model)

    input_tensor = torch.randn(1, 1, 28, 28) 
    output = model(input_tensor)
    print(output.shape) 

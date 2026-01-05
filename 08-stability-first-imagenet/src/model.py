import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    """ResNet-based модель для ImageNet с разделением на backbone и head"""
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        # Используем предобученный ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Backbone: все слои кроме последнего FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Head: новый классификатор
        # ResNet18 имеет 512 выходных каналов после avgpool
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Backbone
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Head
        x = self.fc(x)
        return x

class SimpleImageNetCNN(nn.Module):
    """Упрощенная CNN для ImageNet (если нет предобученной модели)"""
    def __init__(self, num_classes=1000):
        super().__init__()
        # Первый блок
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Второй блок
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Третий блок
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Четвертый блок
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        
        # Backbone FC
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Head
        self.fc3 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 224x224 -> 112x112
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # 112x112 -> 56x56
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        
        # 56x56 -> 28x28
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        
        # 28x28 -> 14x14 -> 1x1
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Backbone
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Head
        x = self.fc3(x)
        return x






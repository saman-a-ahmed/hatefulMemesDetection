import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CNNImageProcessor(nn.Module):
    def __init__(self, output_dim=None, dropout=0.5):
        """
        Custom CNN image processor
        
        Args:
            output_dim (int, optional): Dimension of output. If None, returns features before final layer
            dropout (float): Dropout probability
        """
        super(CNNImageProcessor, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm4 = nn.BatchNorm2d(256)
        
        # Calculate feature dimension after convolutions
        # Input: 224x224x3
        # After 4 pooling layers (stride 2): 14x14x256
        feature_dim = 14 * 14 * 256
        
        # Fully connected layers
        self.fc1 = nn.Linear(feature_dim, 512)
        self.dropout1 = nn.Dropout(dropout)
        
        # Output layer (if specified)
        if output_dim is not None:
            self.fc2 = nn.Linear(512, output_dim)
        else:
            self.fc2 = None
        
        # Save config
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input image tensor [batch_size, 3, height, width]
            return_features (bool): If True, return features before final layer
            
        Returns:
            torch.Tensor: Model output
        """
        # Apply convolutional layers
        x = F.relu(self.norm1(self.pool1(self.conv1(x))))  # [batch_size, 32, 112, 112]
        x = F.relu(self.norm2(self.pool2(self.conv2(x))))  # [batch_size, 64, 56, 56]
        x = F.relu(self.norm3(self.pool3(self.conv3(x))))  # [batch_size, 128, 28, 28]
        x = F.relu(self.norm4(self.pool4(self.conv4(x))))  # [batch_size, 256, 14, 14]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [batch_size, 14*14*256]
        
        # Apply fully connected layer
        features = F.relu(self.fc1(x))
        features = self.dropout1(features)
        
        # Return features if requested
        if return_features:
            return features
        
        # Apply final linear layer if specified
        if self.fc2 is not None:
            return self.fc2(features)
        else:
            return features


class ResNetImageProcessor(nn.Module):
    def __init__(self, output_dim=None, dropout=0.5, pretrained=True, freeze_backbone=False):
        """
        ResNet-based image processor
        
        Args:
            output_dim (int, optional): Dimension of output. If None, returns features before final layer
            dropout (float): Dropout probability
            pretrained (bool): Whether to use pre-trained weights
            freeze_backbone (bool): Whether to freeze backbone parameters
        """
        super(ResNetImageProcessor, self).__init__()
        
        # Load pre-trained ResNet
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        feature_dim = self.resnet.fc.in_features
        
        # Replace final fully connected layer
        self.resnet.fc = nn.Identity()  # Remove classification layer
        
        # Add own classifier
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (if specified)
        if output_dim is not None:
            self.fc = nn.Linear(feature_dim, output_dim)
        else:
            self.fc = None
        
        # Save config
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input image tensor [batch_size, 3, height, width]
            return_features (bool): If True, return features before final layer
            
        Returns:
            torch.Tensor: Model output
        """
        # Extract features using ResNet
        features = self.resnet(x)
        
        # Apply dropout
        features = self.dropout(features)
        
        # Return features if requested
        if return_features:
            return features
        
        # Apply final linear layer if specified
        if self.fc is not None:
            return self.fc(features)
        else:
            return features
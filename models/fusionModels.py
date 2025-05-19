import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusionModel(nn.Module):
    def __init__(self, text_model, image_model, output_dim=1, dropout=0.5):
        """
        Late fusion model that combines text and image models
        
        Args:
            text_model (nn.Module): Text processing model
            image_model (nn.Module): Image processing model
            output_dim (int): Output dimension (1 for binary classification)
            dropout (float): Dropout probability
        """
        super(LateFusionModel, self).__init__()
        
        self.text_model = text_model
        self.image_model = image_model
        
        # Get feature dimensions
        if hasattr(text_model, 'output_dim') and text_model.output_dim is not None:
            self.text_dim = text_model.output_dim
        else:
            # Default dimensions
            self.text_dim = 768 if 'BERT' in text_model.__class__.__name__ else 256
            
        if hasattr(image_model, 'output_dim') and image_model.output_dim is not None:
            self.image_dim = image_model.output_dim
        else:
            # Default dimensions
            self.image_dim = 2048 if 'ResNet' in image_model.__class__.__name__ else 512
        
        # Combined dimension
        combined_dim = self.text_dim + self.image_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, output_dim)
        )
        
    def forward(self, text_input, image_input):
        """
        Forward pass
        
        Args:
            text_input: Input for text model (format depends on text model)
            image_input: Input for image model (tensor of shape [batch_size, 3, height, width])
            
        Returns:
            torch.Tensor: Model output logits
        """
        # Process text
        if isinstance(text_input, dict):
            # BERT input format
            text_features = self.text_model(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                return_features=True
            )
        else:
            # LSTM input format
            text_features = self.text_model(text_input, return_features=True)
        
        # Process image
        image_features = self.image_model(image_input, return_features=True)
        
        # Concatenate features
        combined_features = torch.cat((text_features, image_features), dim=1)
        
        # Apply classifier
        logits = self.classifier(combined_features)
        
        return logits


class EarlyFusionModel(nn.Module):
    def __init__(self, text_model, image_model, output_dim=1, 
                 hidden_dim=512, dropout=0.5, use_attention=False):
        """
        Early fusion model that combines text and image features
        
        Args:
            text_model (nn.Module): Text processing model
            image_model (nn.Module): Image processing model
            output_dim (int): Output dimension (1 for binary classification)
            hidden_dim (int): Hidden dimension for fusion layers
            dropout (float): Dropout probability
            use_attention (bool): Whether to use cross-modal attention
        """
        super(EarlyFusionModel, self).__init__()
        
        self.text_model = text_model
        self.image_model = image_model
        self.use_attention = use_attention
        
        # Get feature dimensions
        if hasattr(text_model, 'output_dim') and text_model.output_dim is not None:
            self.text_dim = text_model.output_dim
        else:
            # Default dimensions
            self.text_dim = 768 if 'BERT' in text_model.__class__.__name__ else 256
            
        if hasattr(image_model, 'output_dim') and image_model.output_dim is not None:
            self.image_dim = image_model.output_dim
        else:
            # Default dimensions
            self.image_dim = 2048 if 'ResNet' in image_model.__class__.__name__ else 512
        
        # Combined dimension
        combined_dim = self.text_dim + self.image_dim
        
        # Cross-modal attention (optional)
        if use_attention:
            self.text_attn = nn.Linear(self.text_dim, 1)
            self.image_attn = nn.Linear(self.image_dim, 1)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, text_input, image_input):
        """
        Forward pass
        
        Args:
            text_input: Input for text model (format depends on text model)
            image_input: Input for image model (tensor of shape [batch_size, 3, height, width])
            
        Returns:
            torch.Tensor: Model output logits
        """
        # Process text
        if isinstance(text_input, dict):
            # BERT input format
            text_features = self.text_model(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                return_features=True
            )
        else:
            # LSTM input format
            text_features = self.text_model(text_input, return_features=True)
        
        # Process image
        image_features = self.image_model(image_input, return_features=True)
        
        # Apply cross-modal attention if enabled
        if self.use_attention:
            # Compute attention scores
            text_score = torch.sigmoid(self.text_attn(text_features))
            image_score = torch.sigmoid(self.image_attn(image_features))
            
            # Normalize scores
            attn_sum = text_score + image_score
            text_weight = text_score / attn_sum
            image_weight = image_score / attn_sum
            
            # Apply attention
            text_features = text_features * text_weight
            image_features = image_features * image_weight
        
        # Concatenate features
        combined_features = torch.cat((text_features, image_features), dim=1)
        
        # Apply fusion layers
        logits = self.fusion(combined_features)
        
        return logits
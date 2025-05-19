
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import numpy as np

class LSTMTextProcessor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, 
                 num_layers=2, output_dim=None, dropout=0.3, 
                 embedding_matrix=None, bidirectional=True):
        """
        LSTM-based text processor
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embeddings
            hidden_dim (int): Dimension of hidden state
            num_layers (int): Number of LSTM layers
            output_dim (int, optional): Dimension of output. If None, returns last hidden state
            dropout (float): Dropout probability
            embedding_matrix (np.ndarray, optional): Pre-trained embedding matrix
            bidirectional (bool): Whether to use bidirectional LSTM
        """
        super(LSTMTextProcessor, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Load pre-trained embeddings if provided
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float))
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (if specified)
        if output_dim is not None:
            lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.fc = nn.Linear(lstm_output_dim, output_dim)
        else:
            self.fc = None
        
        # Save config
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
    def forward(self, text, return_features=False):
        """
        Forward pass
        
        Args:
            text (list or tensor): Input text
                - If list of tokens: Convert to indices first
                - If tensor: Use directly as indices
            return_features (bool): If True, return features before final layer
            
        Returns:
            torch.Tensor: Model output
        """
        # If input is a list of tokens, convert to indices (placeholder - would need vocab mapping)
        if isinstance(text[0], list):
            # This is just a placeholder, would need actual vocab mapping
            text = torch.tensor([[0] * len(tokens) for tokens in text])
        
        # Embed tokens
        embedded = self.embedding(text)
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Get final representation
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states
            # Shape: [batch_size, hidden_dim * 2]
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # Get the final hidden state
            # Shape: [batch_size, hidden_dim]
            hidden = hidden[-1, :, :]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Return features if requested
        if return_features:
            return hidden
        
        # Apply final linear layer if specified
        if self.fc is not None:
            return self.fc(hidden)
        else:
            return hidden


class BERTTextProcessor(nn.Module):
    def __init__(self, output_dim=None, dropout=0.1, freeze_bert=False):
        """
        BERT-based text processor
        
        Args:
            output_dim (int, optional): Dimension of output. If None, returns [CLS] embedding
            dropout (float): Dropout probability
            freeze_bert (bool): Whether to freeze BERT parameters
        """
        super(BERTTextProcessor, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (if specified)
        if output_dim is not None:
            self.fc = nn.Linear(768, output_dim)  # BERT hidden size is 768
        else:
            self.fc = None
        
        # Save config
        self.output_dim = output_dim
        
    def forward(self, input_ids, attention_mask, return_features=False):
        """
        Forward pass
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            return_features (bool): If True, return features before final layer
            
        Returns:
            torch.Tensor: Model output
        """
        # Pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get [CLS] token embedding
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Return features if requested
        if return_features:
            return pooled_output
        
        # Apply final linear layer if specified
        if self.fc is not None:
            return self.fc(pooled_output)
        else:
            return pooled_output
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional


class Attention(nn.Module):
    def __init__(self, lstm_units):
        super(Attention, self).__init__()
        lstm_output_dim = lstm_units * 2
        self.attention_Wa = nn.Linear(lstm_output_dim, lstm_output_dim)
        self.attention_v = nn.Linear(lstm_output_dim, 1, bias=False)

    def forward(self, bilstm_output, key_padding_mask=None):
        mu_w = torch.tanh(self.attention_Wa(bilstm_output))
        scores = self.attention_v(mu_w)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(-1), -float('inf'))

        alpha_w = F.softmax(scores, dim=1)

        context_vector = torch.sum(alpha_w * bilstm_output, dim=1)
        return context_vector, alpha_w


class EnhancedCNN_BiLSTM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 lstm_units: int,
                 cnn_filters: int,
                 cnn_kernel_sizes: List[int],
                 dense_hidden_units: int,
                 engineered_features_dim: int,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.2,
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 attention_type: str = 'paper',
                 ):
        super(EnhancedCNN_BiLSTM, self).__init__()

        self.engineered_features_dim = engineered_features_dim
        self.attention_type = attention_type

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_filters, kernel_size=k)
            for k in cnn_kernel_sizes
        ])
        self.cnn_pool = nn.AdaptiveMaxPool1d(1)
        self.cnn_dropout = nn.Dropout(dropout_rate)
        cnn_output_dim = cnn_filters * len(cnn_kernel_sizes)

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=lstm_units,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=lstm_dropout if lstm_layers > 1 else 0)
        lstm_output_dim = lstm_units * 2
        self.lstm_norm = nn.LayerNorm(lstm_output_dim)

        if attention_type == 'paper':
            self.attention = Attention(lstm_units)
            attention_output_dim = lstm_output_dim
        else:
            # Placeholder for potential future implementation
            pass
        self.attention_dropout = nn.Dropout(dropout_rate)

        combined_input_dim = cnn_output_dim + attention_output_dim + engineered_features_dim
        self.fc1 = nn.Linear(combined_input_dim, dense_hidden_units)
        self.mlp_norm = nn.LayerNorm(dense_hidden_units)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.mlp_dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_hidden_units, 1)

    def forward(self, text_input: torch.Tensor, engineered_features: Optional[torch.Tensor] = None) -> torch.Tensor:

        embedded_words = self.embedding(text_input)

        cnn_input = embedded_words.permute(0, 2, 1)
        cnn_outputs = []
        for conv_layer in self.cnn_layers:
            convolved = F.relu(conv_layer(cnn_input))
            pooled = self.cnn_pool(convolved)
            squeezed = torch.squeeze(pooled, dim=-1)
            dropped = self.cnn_dropout(squeezed)
            cnn_outputs.append(dropped)
        cnn_final_output = torch.cat(cnn_outputs, dim=1)

        lstm_output, (hidden, cell) = self.lstm(embedded_words)
        lstm_output_norm = self.lstm_norm(lstm_output)

        padding_mask = (text_input == self.embedding.padding_idx)
        if self.attention_type == 'paper':
            context_vector, _ = self.attention(lstm_output_norm, key_padding_mask=padding_mask)
        else:
            # Placeholder for potential future implementation
            pass

        attention_final_output = self.attention_dropout(context_vector)

        features_to_merge = [cnn_final_output, attention_final_output]
        if self.engineered_features_dim > 0 and engineered_features is not None:
            features_to_merge.append(engineered_features)

        merged = torch.cat(features_to_merge, dim=1)

        fc1_output = self.fc1(merged)
        normed = self.mlp_norm(fc1_output)
        activated = self.activation(normed)
        dropped = self.mlp_dropout(activated)
        output_logits = self.fc2(dropped)

        return output_logits
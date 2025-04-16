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
        return context_vector


class CNN_BiLSTM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 lstm_units: int,
                 cnn_filters: int,
                 cnn_kernel_size: int,
                 dense_hidden_units: int,
                 engineered_features_dim: int = 0,
                 dropout_rate: float = 0.3
                 ):
        super(CNN_BiLSTM, self).__init__()

        self.engineered_features_dim = engineered_features_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.cnn_layer = nn.Conv1d(in_channels=embedding_dim,
                                   out_channels=cnn_filters,
                                   kernel_size=cnn_kernel_size)
        self.cnn_pool = nn.AdaptiveMaxPool1d(1)
        self.cnn_dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=embedding_dim,
                              hidden_size=lstm_units,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        lstm_output_dim = lstm_units * 2

        self.attention = Attention(lstm_units)
        self.attention_dropout = nn.Dropout(dropout_rate)

        combined_input_dim = cnn_filters + lstm_output_dim + engineered_features_dim
        self.fc1 = nn.Linear(combined_input_dim, dense_hidden_units)
        self.mlp_dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_hidden_units, 1)

    def forward(self, text_input: torch.Tensor, engineered_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        embedded_words = self.embedding(text_input)

        cnn_input = embedded_words.permute(0, 2, 1)
        cnn_convolved = F.relu(self.cnn_layer(cnn_input))
        cnn_pooled = self.cnn_pool(cnn_convolved)
        cnn_output_squeezed = torch.squeeze(cnn_pooled, dim=-1)
        cnn_output = self.cnn_dropout(cnn_output_squeezed)

        bilstm_output, _ = self.bilstm(embedded_words)

        padding_mask = (text_input == self.embedding.padding_idx)
        context_vector = self.attention(bilstm_output, key_padding_mask=padding_mask)
        attention_output = self.attention_dropout(context_vector)

        features_to_merge = [cnn_output, attention_output]
        if self.engineered_features_dim > 0 and engineered_features is not None:
            features_to_merge.append(engineered_features)

        merged = torch.cat(features_to_merge, dim=1)

        hidden = F.relu(self.fc1(merged))
        hidden_dropped = self.mlp_dropout(hidden)
        output = self.fc2(hidden_dropped)

        return output

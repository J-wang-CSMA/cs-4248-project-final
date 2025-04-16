import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_Wa = nn.Linear(feature_dim, feature_dim)
        self.attention_v = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, sequence_output, mask=None):
        mu_w = torch.tanh(self.attention_Wa(sequence_output))
        scores = self.attention_v(mu_w)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1), -float('inf'))
        alpha_w = F.softmax(scores, dim=1)
        context_vector = torch.sum(alpha_w * sequence_output, dim=1)
        return context_vector


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerGRUModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 gru_units,
                 cnn_filters,
                 cnn_kernel_size,
                 dense_hidden_units,
                 nhead,
                 num_encoder_layers,
                 dim_feedforward,
                 engineered_features_dim=0,
                 dropout=0.1
                 ):
        super(TransformerGRUModel, self).__init__()

        self.engineered_features_dim = engineered_features_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cnn_layer = nn.Conv1d(in_channels=embedding_dim,
                                   out_channels=cnn_filters,
                                   kernel_size=cnn_kernel_size)
        self.cnn_pool = nn.AdaptiveMaxPool1d(1)
        self.cnn_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=gru_units,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        gru_output_dim = gru_units * 2
        self.gru_norm = nn.LayerNorm(gru_output_dim)

        self.final_attention = Attention(gru_output_dim)
        self.attention_dropout = nn.Dropout(dropout)

        combined_input_dim = cnn_filters + gru_output_dim + engineered_features_dim
        self.fc1 = nn.Linear(combined_input_dim, dense_hidden_units)
        self.mlp_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_hidden_units, 1)

    def forward(self, text_input, engineered_features=None):

        if self.engineered_features_dim == 0 and engineered_features is not None:
            engineered_features = None

        src_key_padding_mask = (text_input == self.embedding.padding_idx)

        embedded_words = self.embedding(text_input) * math.sqrt(self.embedding_dim)

        cnn_input = embedded_words.permute(0, 2, 1)
        cnn_convolved = F.relu(self.cnn_layer(cnn_input))
        cnn_pooled = self.cnn_pool(cnn_convolved)
        cnn_output_squeezed = torch.squeeze(cnn_pooled, dim=-1)
        cnn_output = self.cnn_dropout(cnn_output_squeezed)

        pos_encoded_input = embedded_words.permute(1, 0, 2)
        pos_encoded = self.pos_encoder(pos_encoded_input)
        pos_encoded_batch_first = pos_encoded.permute(1, 0, 2)

        transformer_output = self.transformer_encoder(pos_encoded_batch_first,
                                                      src_key_padding_mask=src_key_padding_mask)

        gru_output, _ = self.gru(transformer_output)
        gru_output_norm = self.gru_norm(gru_output)

        attended_gru_output = self.final_attention(gru_output_norm, mask=src_key_padding_mask)
        attention_output = self.attention_dropout(attended_gru_output)

        features_to_merge = [cnn_output, attention_output]
        if engineered_features is not None:
            features_to_merge.append(engineered_features)

        merged = torch.cat(features_to_merge, dim=1)

        hidden = F.relu(self.fc1(merged))
        hidden_dropped = self.mlp_dropout(hidden)
        output = self.fc2(hidden_dropped)

        return output

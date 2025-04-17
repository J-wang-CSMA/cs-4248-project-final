import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Any
from transformers import BertModel, AutoTokenizer, BertTokenizer


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(GCNLayer, self).__init__()
        self.linear_a = nn.Linear(in_features, out_features)
        self.linear_d = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj = adj.to(x.device)
        if x.shape[0] == 0:
            out_features = self.linear_d.out_features
            return torch.zeros((0, out_features), device=x.device, dtype=x.dtype)
        if x.shape[0] != adj.shape[0]:
            min_dim = min(x.shape[0], adj.shape[0])
            adj = adj[:min_dim, :min_dim]
            if x.shape[0] > min_dim:
                x = x[:min_dim, :]
            elif x.shape[0] < min_dim:
                out_features = self.linear_d.out_features
                return torch.zeros((x.shape[0], out_features), device=x.device, dtype=x.dtype)

        transformed_features_a = self.linear_a(x)
        convolved_features = torch.matmul(adj, transformed_features_a)
        activated_features = self.relu(convolved_features)
        transformed_features_d = self.linear_d(activated_features)
        output = self.relu(transformed_features_d)
        return output


class EnhancedBertGCNModel(nn.Module):
    def __init__(self, bert_model_name: str = 'bert-base-uncased',
                 gcn_layers: int = 4,
                 num_handcrafted_features: int = 0,
                 classifier_hidden_dims: List[int] = [1024, 256],
                 dropout_rate: float = 0.5,
                 text_dropout: float = 0.3,
                 num_classes: int = 2,
                 freeze_bert: bool = True):
        super(EnhancedBertGCNModel, self).__init__()
        self.bert_model_name = bert_model_name
        self.num_handcrafted_features = num_handcrafted_features
        self.num_classes = num_classes
        self.num_gcn_layers = gcn_layers

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.gcn_layers = nn.ModuleList([
            GCNLayer(self.bert_hidden_size, self.bert_hidden_size) for _ in range(gcn_layers)
        ])

        self.text_embed_dropout = nn.Dropout(text_dropout)

        classifier_input_dim = self.bert_hidden_size + num_handcrafted_features
        mlp_layers = []
        current_dim = classifier_input_dim
        for hidden_dim in classifier_hidden_dims:
            mlp_layers.append(nn.Linear(current_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        mlp_layers.append(nn.Linear(current_dim, num_classes))

        self.classifier = nn.Sequential(*mlp_layers)

    def forward(self,
                aligned_bert_embeddings: torch.Tensor,
                dependency_graph: torch.Tensor,
                affective_graph: torch.Tensor,
                handcrafted_features: torch.Tensor) -> torch.Tensor:

        if aligned_bert_embeddings is None or aligned_bert_embeddings.shape[0] == 0:
            return torch.zeros(self.num_classes,
                               device=aligned_bert_embeddings.device if aligned_bert_embeddings is not None else 'cpu')

        num_nodes = aligned_bert_embeddings.shape[0]
        if dependency_graph.shape[0] != num_nodes:
            dependency_graph = dependency_graph[:num_nodes, :num_nodes]
        if affective_graph.shape[0] != num_nodes:
            affective_graph = affective_graph[:num_nodes, :num_nodes]

        node_features_initial = aligned_bert_embeddings.to(dependency_graph.device)
        dep_adj = dependency_graph
        aff_adj = affective_graph
        handcrafted_features_tensor = handcrafted_features.to(node_features_initial.device)

        gcn_output = None
        current_features = node_features_initial

        for i, layer in enumerate(self.gcn_layers):
            if i % 2 == 0:
                input_feat = node_features_initial
                adj_matrix = dep_adj
            else:
                if gcn_output is None:
                    gcn_output = node_features_initial
                input_feat = gcn_output
                adj_matrix = aff_adj

            if adj_matrix.shape[0] != input_feat.shape[0]:
                min_dim = min(adj_matrix.shape[0], input_feat.shape[0])
                if min_dim == 0:
                    return torch.zeros(self.num_classes, device=node_features_initial.device)
                adj_matrix = adj_matrix[:min_dim, :min_dim]
                input_feat = input_feat[:min_dim, :]

            gcn_output = layer(input_feat, adj_matrix)

        final_gcn_output = gcn_output
        if final_gcn_output is None or final_gcn_output.shape[0] == 0:
            return torch.zeros(self.num_classes, device=node_features_initial.device)

        alpha_mat = torch.matmul(final_gcn_output, node_features_initial.transpose(0, 1))
        alpha = F.softmax(alpha_mat.sum(0, keepdim=True), dim=1)
        aggregated_embedding = torch.matmul(alpha, node_features_initial).squeeze(0)

        aggregated_embedding_dropped = self.text_embed_dropout(aggregated_embedding)

        if handcrafted_features_tensor.dim() != 1:
            handcrafted_features_tensor = handcrafted_features_tensor.flatten()
        if handcrafted_features_tensor.shape[0] != self.num_handcrafted_features:
            handcrafted_features_tensor = torch.zeros(self.num_handcrafted_features,
                                                      device=node_features_initial.device)

        combined_embedding = torch.cat((aggregated_embedding_dropped, handcrafted_features_tensor), dim=0)

        logits = self.classifier(combined_embedding)

        return logits

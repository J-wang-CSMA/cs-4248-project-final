import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer, AutoTokenizer
import numpy as np
from typing import Tuple, List, Dict, Optional, Union


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(GCNLayer, self).__init__()
        self.linear_a = nn.Linear(in_features, out_features)
        self.linear_d = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj = adj.to(x.device)
        if x.shape[0] != adj.shape[0]:
            # Simplified error handling: return input or zero tensor
            return torch.zeros_like(self.linear_d(self.relu(self.linear_a(x))))

        transformed_features_a = self.linear_a(x)
        convolved_features = torch.matmul(adj, transformed_features_a)
        activated_features = self.relu(convolved_features)
        transformed_features_d = self.linear_d(activated_features)
        output = self.relu(transformed_features_d)
        return output


class BertGCNModel(nn.Module):
    def __init__(self, bert_model_name: str = 'bert-base-uncased',
                 gcn_hidden_dim: int = 300, gcn_layers: int = 4, num_classes: int = 2,
                 freeze_bert: bool = True):
        super(BertGCNModel, self).__init__()
        self.bert_model_name = bert_model_name
        self.gcn_hidden_dim = gcn_hidden_dim
        self.num_classes = num_classes

        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Note: Tokenizer should be initialized outside and potentially passed if needed
        # self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True) # Or BertTokenizer

        self.fc = nn.Linear(bert_hidden_size, gcn_hidden_dim)

        self.gcn_layers = nn.ModuleList([
            GCNLayer(gcn_hidden_dim, gcn_hidden_dim) for _ in range(gcn_layers)
        ])

        self.classifier = nn.Linear(gcn_hidden_dim, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def robust_align_bert_spacy(self, sentence: str, bert_outputs: object, spacy_tokens: List[str], tokenizer, nlp) -> \
    Optional[torch.Tensor]:
        try:
            bert_embeddings = bert_outputs.last_hidden_state.squeeze(0).to(self.device)
            encoding = tokenizer(sentence, return_tensors='pt', truncation=True,
                                 max_length=tokenizer.model_max_length,
                                 return_offsets_mapping=True)
            if 'offset_mapping' not in encoding:
                return None
            offset_mapping = encoding['offset_mapping'].squeeze(0).cpu().numpy()
            bert_input_ids = encoding['input_ids'].squeeze(0)
            num_bert_tokens = bert_input_ids.shape[0]

            if bert_embeddings.shape[0] != num_bert_tokens:
                bert_embeddings = bert_embeddings[:num_bert_tokens, :]
                if bert_embeddings.shape[0] != num_bert_tokens:
                    return None
        except Exception:
            return None

        num_spacy_tokens = len(spacy_tokens)
        if num_spacy_tokens == 0: return None

        try:
            doc = nlp(sentence)
            if len(doc) != num_spacy_tokens:
                return None
            spacy_token_spans = [(token.idx, token.idx + len(token.text)) for token in doc]
        except Exception:
            return None

        aligned_embeddings = []
        bert_token_idx = 0

        for spacy_idx in range(num_spacy_tokens):
            spacy_start, spacy_end = spacy_token_spans[spacy_idx]
            current_spacy_token_bert_indices = []
            temp_bert_idx = bert_token_idx

            while temp_bert_idx < len(offset_mapping):
                bert_start, bert_end = offset_mapping[temp_bert_idx]
                if bert_start == 0 and bert_end == 0:
                    temp_bert_idx += 1
                    continue
                has_overlap = max(spacy_start, bert_start) < min(spacy_end, bert_end)
                if has_overlap:
                    current_spacy_token_bert_indices.append(temp_bert_idx)
                    temp_bert_idx += 1
                    if bert_end >= spacy_end:
                        break
                elif bert_start >= spacy_end:
                    break
                else:
                    temp_bert_idx += 1

            bert_token_idx = temp_bert_idx

            if current_spacy_token_bert_indices:
                valid_indices = [idx for idx in current_spacy_token_bert_indices if idx < bert_embeddings.shape[0]]
                if valid_indices:
                    token_embeddings = bert_embeddings[valid_indices, :]
                    avg_embedding = torch.mean(token_embeddings, dim=0)
                    aligned_embeddings.append(avg_embedding)
                else:
                    aligned_embeddings.append(torch.zeros(self.bert.config.hidden_size, device=self.device))
            else:
                aligned_embeddings.append(torch.zeros(self.bert.config.hidden_size, device=self.device))

        if not aligned_embeddings:
            return None
        try:
            aligned_tensor = torch.stack(aligned_embeddings)
        except RuntimeError:
            return None

        if aligned_tensor.shape[0] != num_spacy_tokens:
            return None
        return aligned_tensor

    def forward(self, sentence: str, tokenizer, nlp, build_dependency_graph_func,
                build_affective_graph_func) -> torch.Tensor:
        if not isinstance(sentence, str) or not sentence.strip():
            return torch.zeros(self.num_classes, device=self.device)

        spacy_tokens, dep_adj = build_dependency_graph_func(sentence, nlp)
        if not spacy_tokens:
            return torch.zeros(self.num_classes, device=self.device)
        aff_adj = build_affective_graph_func(spacy_tokens)
        num_spacy_tokens = len(spacy_tokens)

        try:
            encoding = tokenizer(sentence, return_tensors='pt', truncation=True,
                                 max_length=tokenizer.model_max_length, padding=True)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            with torch.no_grad():
                bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        except Exception:
            return torch.zeros(self.num_classes, device=self.device)

        bert_aligned_embeddings = self.robust_align_bert_spacy(sentence, bert_outputs, spacy_tokens, tokenizer, nlp)

        if bert_aligned_embeddings is None or bert_aligned_embeddings.shape[0] != num_spacy_tokens:
            return torch.zeros(self.num_classes, device=self.device)

        try:
            gcn_input_features = self.fc(bert_aligned_embeddings)
        except Exception:
            return torch.zeros(self.num_classes, device=self.device)

        if dep_adj.size(0) != num_spacy_tokens or aff_adj.size(0) != num_spacy_tokens:
            dep_adj = dep_adj[:num_spacy_tokens, :num_spacy_tokens]
            aff_adj = aff_adj[:num_spacy_tokens, :num_spacy_tokens]

        current_features = gcn_input_features
        try:
            for i, layer in enumerate(self.gcn_layers):
                adj_matrix = dep_adj if i % 2 == 0 else aff_adj
                current_features = layer(current_features, adj_matrix)
        except Exception:
            return torch.zeros(self.num_classes, device=self.device)

        if current_features.shape[0] > 0:
            final_graph_embedding = torch.mean(current_features, dim=0)
        else:
            return torch.zeros(self.num_classes, device=self.device)

        try:
            logits = self.classifier(final_graph_embedding)
        except Exception:
            return torch.zeros(self.num_classes, device=self.device)

        return logits

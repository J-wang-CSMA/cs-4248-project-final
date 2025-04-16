import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from typing import Tuple, List, Dict, Union, Optional, Any


class BertWithFeaturesConfig(BertConfig):
    def __init__(self, handcrafted_feature_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.handcrafted_feature_dim = handcrafted_feature_dim


class BertWithFeaturesClassifier(BertPreTrainedModel):
    config_class = BertWithFeaturesConfig

    def __init__(self, config: BertWithFeaturesConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.handcrafted_feature_dim = config.handcrafted_feature_dim

        self.bert = BertModel(config, add_pooling_layer=True)

        classifier_dropout_prob = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        combined_dim = config.hidden_size + self.handcrafted_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.ReLU(),
            nn.Dropout(classifier_dropout_prob),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout_prob),
            nn.Linear(256, config.num_labels)
        )

        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            handcrafted_features: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        combined_features = torch.cat((pooled_output, handcrafted_features.float()), dim=1)

        logits = self.classifier(combined_features)

        loss = None

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return {
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.activations import get_activation
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional

class BinaryClassificationModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.ko_electra = ElectraModel(config)
        self.classifier = BinaryElectraClassification(config)
        # for name, param in self.ko_electra.named_parameters():
        #     param.requires_grad = False

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = self.ko_electra(
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
        pooled_output = hidden_states[0]

        logits = self.classifier(pooled_output)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (logits,) + hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states.hidden_states,
            attentions=hidden_states.attentions,
        )

class BinaryElectraClassification(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.words = args.maxlen
        self.embedding_size = 768
        self.filters = args.maxlen // 2
        self.kernel_h = [3, 4, 5]

        self.cnn_1d_1 = nn.Conv2d(in_channels=1,
                                  kernel_size=(self.kernel_h[0], self.embedding_size),
                                  out_channels=self.filters)
        self.cnn_1d_2 = nn.Conv2d(in_channels=1,
                                  kernel_size=(self.kernel_h[1], self.embedding_size),
                                  out_channels=self.filters)
        self.cnn_1d_3 = nn.Conv2d(in_channels=1,
                                  kernel_size=(self.kernel_h[2], self.embedding_size),
                                  out_channels=self.filters)

        self.max_pool_1 = nn.MaxPool1d(self.words - self.kernel_h[0])
        self.max_pool_2 = nn.MaxPool1d(self.words - self.kernel_h[1])
        self.max_pool_3 = nn.MaxPool1d(self.words - self.kernel_h[2])

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # features: [32, 512, 768], [batch size, embedded size, hidden size]
        # features = torch.transpose(features, 1, 2)
        features = torch.unsqueeze(features, 1) # [32, 1, 512, 768]

        conv1 = F.gelu(self.cnn_1d_1(features)) # [32, 256, 510]
        conv2 = F.gelu(self.cnn_1d_2(features)) # [32, 256, 509]
        conv3 = F.gelu(self.cnn_1d_3(features)) # [32, 256, 508]

        conv1 = torch.squeeze(conv1, 3) # [32, 256, 510, 1]
        conv2 = torch.squeeze(conv2, 3) # [32, 256, 509, 1]
        conv3 = torch.squeeze(conv3, 3) # [32, 256, 508, 1]

        pool_1 = self.max_pool_1(conv1) # [32, 256, 1]
        pool_2 = self.max_pool_2(conv2) # [32, 256, 1]
        pool_3 = self.max_pool_3(conv3) # [32, 256, 1]

        x = torch.cat([pool_1, pool_2, pool_3], 2)
        x = torch.flatten(x, 1)

        # x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        # return shape: [32, 2]
        return x
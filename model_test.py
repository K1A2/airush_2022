from transformers import AutoTokenizer, ElectraForSequenceClassification, AutoModel
import torch.nn as nn
import pytorch_model_summary

model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")

for param in model.electra.parameters():
    param.require_grad = False

# model.classifier = nn.Sequential(
#     nn.Linear(in_features=768, out_features=768, bias=True),
#     nn.LeakyReLU(),
#     nn.Linear(in_features=768, out_features=768, bias=True),
#     nn.LayerNorm(768),
#     nn.Dropout(0.1),
#     nn.LeakyReLU(),
#     nn.Linear(in_features=768, out_features=384, bias=True),
#     nn.LeakyReLU(),
#     nn.Linear(in_features=384, out_features=192, bias=True),
#     nn.LayerNorm(192),
#     nn.Dropout(0.1),
#     nn.LeakyReLU(),
#     nn.Linear(in_features=192, out_features=2, bias=True),
#     nn.Sigmoid(),
# )

print(model)

print(pytorch_model_summary.summary(model, ))
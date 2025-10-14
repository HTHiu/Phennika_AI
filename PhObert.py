import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from underthesea import word_tokenize

try:
    def vi_segment(texts):
        if isinstance(texts, str):
            texts = [texts]
        return [" ".join(word_tokenize(t, format="text")) for t in texts]
except Exception:
    def vi_segment(texts):
        return texts if isinstance(texts, list) else [texts]

class TextEncoder(nn.Module):
    def __init__(self, model_name="vinai/phobert-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, text):
        
        text = vi_segment(text)
        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        outputs = self.model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])

        cls = outputs.last_hidden_state[:, 0, :]
        return cls  

class MLPb(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", proj_dim=512, num_classes=6, dropout=0.1, freeze_encoder=False):
        super().__init__()
        self.text_encoder = TextEncoder(model_name=model_name)
        hidden = self.text_encoder.model.config.hidden_size  

        if freeze_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.text_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.final_fc = nn.Linear(proj_dim, num_classes)

    def forward(self, text):
        x = self.text_encoder(text)           # [B, H]
        x = self.text_fc(x)                   # [B, proj_dim]
        logits = self.final_fc(x)             # [B, num_classes]
        return logits

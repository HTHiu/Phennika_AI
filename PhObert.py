import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_name="vinai/phobert-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        outputs = self.model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])

        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding

class MLPb(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", freeze_encoder=False):
        super().__init__()
        self.text_encoder = TextEncoder(model_name=model_name)
        if freeze_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.classification_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 6)
        )
        self.sentiment_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 6)
        )

    def forward(self, text):
        shared_embedding = self.text_encoder(text)
        
        relevance_logits = self.classification_head(shared_embedding)
        sentiment_scores = self.sentiment_head(shared_embedding)
        relevance_preds = torch.sigmoid(relevance_logits)

        return {
            "relevance_preds": relevance_preds,  
            "relevance_logits": relevance_logits,
            "sentiment_preds": sentiment_scores  
        }

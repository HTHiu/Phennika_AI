import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name="/kaggle/input/phobert_simcse_finetuned/pytorch/default/1"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        outputs = self.model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])

        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding

class AspectCorrelationHead(nn.Module):
    """
    Module này thay thế cho nn.Sequential cũ.
    Nó sử dụng self-attention để mô hình hóa sự tương quan giữa các khía cạnh.
    """
    def __init__(self, embed_size, num_labels=6, num_heads=8):
        super().__init__()
        self.num_labels = num_labels
        self.embed_size = embed_size
        self.initial_projection = nn.Linear(embed_size, embed_size * num_labels)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.final_projection = nn.Linear(embed_size, 1)

    def forward(self, shared_embedding):
   
        initial_aspect_embeds = self.initial_projection(shared_embedding).view(-1, self.num_labels, self.embed_size)
        
        attn_output, _ = self.attention(initial_aspect_embeds, initial_aspect_embeds, initial_aspect_embeds)
        enriched_aspect_embeds = self.layer_norm(initial_aspect_embeds + attn_output)

        logits = self.final_projection(enriched_aspect_embeds).squeeze(-1)
        
        return logits

class MLPb(nn.Module):
    def __init__(self, model_name="/kaggle/input/phobert_simcse_finetuned/pytorch/default/1", freeze_encoder=False):
        super().__init__()
        self.text_encoder = TextEncoder(model_name=model_name)
        if freeze_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        encoder_hidden_size = 768

        self.classification_head = AspectCorrelationHead(
            embed_size=encoder_hidden_size, 
            num_labels=6, 
            num_heads=8
        )
        # ---------------------------


        self.sentiment_head = nn.Sequential(
            nn.Linear(encoder_hidden_size, 256),
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
            "relevance_logits": relevance_logits,
            "relevance_preds": relevance_preds,
            "sentiment_preds": sentiment_scores  
        }

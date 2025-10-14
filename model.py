# Cell 2: Copy toàn bộ code model (từ import đến main, nhưng loại bỏ if __name__ và parser)
"""
Multi-task model simplified to predict sentiment scores (0-5) per label, where 0 means not mentioned.
"""

import re
import ast
import json
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F

# Optional metrics
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error

# -------------------------
# Config / Hyperparams
# -------------------------
MODEL_NAME = "vinai/phobert-base"
NUM_LABELS = 6
NUM_SENT_CLASSES = 6  # 0-5, 0 means not mentioned
MAX_LEN = 256
BATCH_SIZE = 16  # Increased for better gradient estimates
LR = 1e-5  # Lower LR for stability
EPOCHS = 5  # Increased epochs for better convergence
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DECAY = 1e-2  # Added weight decay
GRAD_CLIP = 1.0  # Gradient clipping

LABEL_NAMES = ["giai_tri","luu_tru","nha_hang","an_uong","van_chuyen","mua_sam"]

# -------------------------
# Keyword mapping and lexicons
# -------------------------
# Map label index (1..6) to keyword list (lowercase phrases)
KEYWORD_MAP = {
    1: ["wifi", "wi-fi", "hồ bơi", "pool", "khu vui chơi", "khu giải trí", "sân chơi", "bãi biển", "tour", "show", "vé tham quan"],
    2: ["khách sạn", "resort", "homestay", "phòng", "check-in", "check out", "room", "villa", "lưu trú", "nhà nghỉ"],
    3: ["nhà hàng", "nhà hàng sang", "ẩm thực", "quán ăn cao cấp", "fine dining", "nhà hàng", "set menu", "chef"],
    4: ["đồ ăn", "thức uống", "món", "ăn uống", "cafe", "quán ăn", "buffet", "bữa sáng"],
    5: ["taxi", "grab", "xe bus", "xe buýt", "tàu", "phương tiện", "giao thông", "sân bay", "đi lại", "xe máy", "thuê xe"],
    6: ["mua sắm", "cửa hàng", "shop", "shopping", "quầy", "trung tâm thương mại", "mall", "giá cả", "ưu đãi"]
}

# Improved sentiment lexicon with scores: +2 strong pos, +1 pos, 0 neutral, -1 neg, -2 strong neg
SENTIMENT_LEXICON = {
    # Strong positive +2
    "tuyệt vời": 2, "hoàn hảo": 2, "xuất sắc": 2, "tuyệt": 2, "siêu": 2,
    # Positive +1
    "tốt": 1, "ngon": 1, "đẹp": 1, "thoải mái": 1, "nhanh": 1, "rẻ": 1, "sạch": 1, "tiện lợi": 1,
    # Neutral 0
    "ổn": 0, "ok": 0, "bình thường": 0, "tạm": 0,
    # Negative -1
    "kém": -1, "tệ": -1, "chậm": -1, "đắt": -1, "xấu": -1, "không tốt": -1, "không ổn": -1,
    # Strong negative -2
    "thất vọng": -2, "hỏng": -2, "bẩn": -2, "kinh khủng": -2, "tồi tệ": -2, "rất kém": -2
}

# We will treat multi-word patterns too; make a compiled regex for each keyword for word-boundary matching
def compile_keyword_patterns(keyword_map):
    compiled = {}
    for lab, kw_list in keyword_map.items():
        patterns = []
        for kw in kw_list:
            # escape special regex chars, match word boundaries (or unicode word boundaries)
            pat = re.compile(r'\b' + re.escape(kw.lower()) + r'\b', flags=re.IGNORECASE | re.UNICODE)
            patterns.append(pat)
        compiled[lab] = patterns
    return compiled

COMPILED_KEYWORD_PATTERNS = compile_keyword_patterns(KEYWORD_MAP)

# -------------------------
# Helper: apply keyword mapping to a DataFrame row
# -------------------------
def apply_keyword_mapping_to_text(text: str,
                                  compiled_patterns,
                                  sentiment_lexicon=SENTIMENT_LEXICON,
                                  window=10) -> Dict[int,int]:
    """
    Returns sentiments_dict mapping label_id -> star (0-5), 0 if not found.
    Improved: Use scored lexicon, sum scores in larger window, map to 1-5 if found, else 0.
    Aggregate per label: average scores from occurrences, then map to star.
    """
    text_low = text.lower()
    tokens = re.findall(r"\w+|[^\s\w]", text_low, flags=re.UNICODE)  # keep punctuation
    # create index mapping from char position to token idx to allow window search by token
    token_positions = []
    start = 0
    for tok in tokens:
        idx = text_low.find(tok, start)
        if idx == -1:
            continue
        token_positions.append((tok, idx))
        start = idx + len(tok)

    score_sums = defaultdict(float)  # label -> sum of scores
    count_occ = defaultdict(int)  # label -> num occurrences

    # For each label, check all patterns
    for lab, patterns in compiled_patterns.items():
        for pat in patterns:
            for m in pat.finditer(text_low):
                # find token index of match start
                char_pos = m.start()
                # find nearest token index
                token_idx = next((i for i, (_, pos) in enumerate(token_positions) if pos >= char_pos), len(token_positions) - 1)
                # gather nearby tokens
                start_idx = max(0, token_idx - window)
                end_idx = min(len(tokens), token_idx + window + 1)
                window_tokens = tokens[start_idx:end_idx]
                # sum lexicon scores in window
                window_score = sum(sentiment_lexicon.get(w, 0) for w in window_tokens)
                score_sums[lab] += window_score
                count_occ[lab] += 1

    # Compute average score per label and map to star
    sentiments = {lab: 0 for lab in range(1, NUM_LABELS + 1)}  # default 0
    for lab in count_occ:
        if count_occ[lab] > 0:
            avg_score = score_sums[lab] / count_occ[lab]
            if avg_score >= 1.5:
                star = 5
            elif avg_score >= 0.5:
                star = 4
            elif avg_score > -0.5:
                star = 3
            elif avg_score > -1.5:
                star = 2
            else:
                star = 1
            sentiments[lab] = star

    return sentiments

def apply_keyword_mapping_row(row, compiled_patterns=COMPILED_KEYWORD_PATTERNS, overwrite=False):
    """
    row: DataFrame row
    overwrite: if True, allow keyword mapping to overwrite existing sentiment values (non-zero)
    Updates:
        - row['s_giai_tri' .. 's_mua_sam'] ints (0-5), 0 means NA or not mentioned
    No 'labels' column anymore.
    """
    text = str(row.get('text', "")).strip()
    if text == "":
        return row
    # parse existing sentiments columns (if present)
    sentiments_cols = ['s_giai_tri','s_luu_tru','s_nha_hang','s_an_uong','s_van_chuyen','s_mua_sam']
    cur_sents = {}
    for idx, col in enumerate(sentiments_cols, start=1):
        v = row.get(col, 0)
        try:
            vi = int(v)
        except:
            vi = 0
        cur_sents[idx] = vi  # 0 means NA or not present

    # apply mapping
    found_sents = apply_keyword_mapping_to_text(text, compiled_patterns)
    # set sentiment if not present or if overwrite True
    for lab in range(1, NUM_LABELS + 1):
        if (cur_sents.get(lab, 0) == 0) or overwrite:
            cur_sents[lab] = found_sents.get(lab, 0)

    # write back
    for idx, col in enumerate(sentiments_cols, start=1):
        row[col] = int(cur_sents.get(idx, 0))
    return row

# -------------------------
# Dataset class
# -------------------------
class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len=MAX_LEN):
        self.texts = df['text'].astype(str).tolist()
        self.sentiments = []  # list of lists length NUM_LABELS, values 0-5
        for _, row in df.iterrows():
            # get per-label sentiments columns if exist
            s_cols = ['s_giai_tri','s_luu_tru','s_nha_hang','s_an_uong','s_van_chuyen','s_mua_sam']
            sent_row = []
            for c in s_cols:
                v = row.get(c, 0)
                try:
                    vi = int(v)
                except:
                    vi = 0
                sent_row.append(vi)  # 0-5
            self.sentiments.append(sent_row)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # sentiments: 0-5 as classes 0-5
        item['sentiments'] = torch.tensor(self.sentiments[idx], dtype=torch.long)  # shape (NUM_LABELS,)
        return item

# -------------------------
# Model
# -------------------------
class MultiTaskModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_LABELS, num_sent_classes=NUM_SENT_CLASSES, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)  # Added dropout for regularization
        self.sent_head = nn.Linear(hidden, num_labels * num_sent_classes)
        self.num_labels = num_labels
        self.num_sent_classes = num_sent_classes

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)  # Changed to mean pooling for better representation
        pooled = self.dropout(pooled)
        sent_logits = self.sent_head(pooled)  # (B, L*C)
        sent_logits = sent_logits.view(-1, self.num_labels, self.num_sent_classes)  # (B, L, C)
        return sent_logits

# -------------------------
# Loss helper
# -------------------------
def compute_loss(sent_logits, sentiments):
    sent_loss_fn = nn.CrossEntropyLoss()  # no ignore_index, since 0 is a valid class
    B, L, C = sent_logits.shape
    sent_logits_flat = sent_logits.view(B*L, C)
    sentiments_flat = sentiments.view(B*L)
    loss_sent = sent_loss_fn(sent_logits_flat, sentiments_flat)
    return loss_sent, loss_sent.detach().item()

# -------------------------
# Train / Eval loops
# -------------------------
def train_epoch(model, dataloader, optimizer, scheduler=None):
    model.train()
    total_loss = 0.0
    total_sent_loss = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        sentiments = batch['sentiments'].to(DEVICE)
        optimizer.zero_grad()
        sent_logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss, l_sent = compute_loss(sent_logits, sentiments)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # Added gradient clipping
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        total_sent_loss += l_sent
    n = len(dataloader)
    return total_loss/n, total_sent_loss/n

def eval_model(model, dataloader):
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    all_true_sents = []
    all_pred_sents = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            sentiments = batch['sentiments'].cpu().numpy()  # 0-5
            sent_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            sent_preds = torch.argmax(sent_logits, dim=-1).cpu().numpy()  # 0-5
            # infer labels: >0
            true_bin = (sentiments > 0).astype(int)
            pred_bin = (sent_preds > 0).astype(int)
            all_true_labels.append(true_bin)
            all_pred_labels.append(pred_bin)
            # collect sentiments where true >0
            B, L = sentiments.shape
            for i in range(B):
                for j in range(L):
                    if sentiments[i, j] > 0:
                        all_true_sents.append(int(sentiments[i, j]))
                        all_pred_sents.append(int(sent_preds[i, j]))
    if len(all_true_labels) == 0:
        return {}
    y_true = np.vstack(all_true_labels)
    y_pred = np.vstack(all_pred_labels)
    # Compute micro F1 for labels
    micro_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    results = {
        'label_micro_f1': float(micro_f1),
        'label_macro_f1': float(macro_f1)
    }
    if len(all_true_sents) > 0:
        mae = mean_absolute_error(all_true_sents, all_pred_sents)
        acc = accuracy_score(all_true_sents, all_pred_sents)
        results.update({'sent_mae': float(mae), 'sent_acc': float(acc)})
    return results

# -------------------------
# Utilities: load data and apply mapping
# -------------------------
def load_and_apply_rules(csv_path: str, apply_rules=True, overwrite=False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Rename 'Review' to 'text'
    if 'Review' in df.columns:
        df = df.rename(columns={'Review': 'text'})
    # Rename sentiment columns to match expected names (remove '# ' or '#' if present)
    rename_dict = {
        '# giai_tri': 's_giai_tri',
        '# luu_tru': 's_luu_tru',
        '# nha_hang': 's_nha_hang',
        '# an_uong': 's_an_uong',
        '# van_chuyen': 's_van_chuyen',
        '# mua_sam': 's_mua_sam'
    }
    df = df.rename(columns=rename_dict)
    # ensure sentiment columns exist
    s_cols = ['s_giai_tri','s_luu_tru','s_nha_hang','s_an_uong','s_van_chuyen','s_mua_sam']
    for c in s_cols:
        if c not in df.columns:
            df[c] = 0
    if apply_rules:
        df = df.apply(lambda r: apply_keyword_mapping_row(r, COMPILED_KEYWORD_PATTERNS, overwrite=overwrite), axis=1)
    # remove 'labels' if exists
    if 'labels' in df.columns:
        df = df.drop(columns=['labels'])
    return df

# -------------------------
# Prediction helper (not needed for training, but keep for completeness)
# -------------------------
def predict_texts(model, tokenizer, texts: List[str]):
    model.eval()
    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)
    with torch.no_grad():
        sent_logits = model(input_ids=input_ids, attention_mask=attention_mask)
        sent_preds = torch.argmax(sent_logits, dim=-1).cpu().numpy()  # 0-5
    outputs = []
    for i, t in enumerate(texts):
        labs = [LABEL_NAMES[j] for j in range(NUM_LABELS) if sent_preds[i, j] > 0]
        sdict = {}
        for j in range(NUM_LABELS):
            pred = int(sent_preds[i, j])
            if pred > 0:
                sdict[LABEL_NAMES[j]] = pred
        outputs.append({'text': t, 'labels': labs, 'sentiments': sdict})
    return outputs

# -------------------------
# Main function (adjusted for notebook)
# -------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    # Set paths for Kaggle
    train_csv = '/kaggle/input/train-model/train.csv'  # Corrected path based on your dataset name
    model_path = '/kaggle/working/best_model.pt'  # Save model in /kaggle/working (downloadable after)
    
    print("Loading training data and applying keyword rules...")
    df = load_and_apply_rules(train_csv, apply_rules=True, overwrite=False)
    # optionally shuffle / split
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(0.1 * n))
    df_train = df.iloc[n_val:]
    df_val = df.iloc[:n_val]
    train_ds = ReviewDataset(df_train, tokenizer, max_len=MAX_LEN)
    val_ds = ReviewDataset(df_val, tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    model = MultiTaskModel().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # Added weight decay
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps)
    best_val = -1.0
    for epoch in range(EPOCHS):
        tr_loss, tr_sent_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_metrics = eval_model(model, val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {tr_loss:.4f}, sent_loss: {tr_sent_loss:.4f}")
        print(f"  Val metrics: {val_metrics}")
        # simple save best on label_micro_f1
        score = val_metrics.get('label_micro_f1', 0.0)
        if score > best_val:
            best_val = score
            torch.save({'model_state': model.state_dict(), 'tokenizer': tokenizer.__dict__}, model_path)
            print(f"  Saved best model to {model_path}")
    print("Training finished.")

# Chạy main
main()

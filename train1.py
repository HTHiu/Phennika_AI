import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam,AdamW
from tqdm import tqdm
import csv
import pandas as pd
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score
LABEL_COLS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]
VALID_SIZE = 0.2
RANDOM_STATE = 42 
full_df = pd.read_csv('/kaggle/input/hackat/train-problem.csv')

label_counts = full_df[LABEL_COLS].astype(str).agg('-'.join, axis=1)

mask = full_df.index.isin(
  label_counts.loc[label_counts.duplicated(keep=False)].index
)

filtered_df = full_df[mask]

labels = filtered_df[LABEL_COLS]
train_df, val_df = train_test_split(
  filtered_df, 
  test_size=0.2, 
  random_state=42,
  stratify=labels
)

train_dataset = MyData(train_df)
val_dataset = MyData(val_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, drop_last=True)

def evaluate(model, dataloader, device):
    """
    Hàm đánh giá mô hình trên tập validation, trả về 3 metrics theo yêu cầu.
    """
    model.eval()  
    all_relevance_preds= [] 
    all_relevance_labels= []
    all_sentiment_preds= []
    all_sentiment_labels= []

    with torch.no_grad(): 
        for batch in dataloader:
            texts = batch["text"]
            relevance_labels = batch["relevance_label"]
            sentiment_labels = batch["sentiment_label"]
            outputs = model(texts)
            
            # Lấy dự đoán và chuyển về CPU để xử lý

            all_relevance_preds.append(outputs['relevance_preds'].cpu())
            all_sentiment_preds.append(outputs['sentiment_preds'].cpu())
            
            all_relevance_labels.append(relevance_labels.cpu())
            all_sentiment_labels.append(sentiment_labels.cpu())

    all_relevance_preds = torch.cat(all_relevance_preds, dim=0)
    all_sentiment_preds = torch.cat(all_sentiment_preds, dim=0)
    all_relevance_labels = torch.cat(all_relevance_labels, dim=0)
    all_sentiment_labels = torch.cat(all_sentiment_labels, dim=0)

    binary_relevance_preds = (all_relevance_preds > 0.5).int()
    
    # Dùng sklearn để tính micro F1 score
    micro_f1 = f1_score(all_relevance_labels.numpy(), binary_relevance_preds.numpy(), average='micro', zero_division=0)

    # --- 2. Tính Sentiment Quality Score ---
    # Tạo mask để chỉ chọn những vị trí mà nhãn thực tế > 0
    sentiment_mask = all_relevance_labels == 1
    
    # Lấy ra các dự đoán và nhãn tương ứng với mask
    preds_to_evaluate = all_sentiment_preds[sentiment_mask]
    labels_to_evaluate = all_sentiment_labels[sentiment_mask]
    
    # Làm tròn dự đoán để tính "độ chính xác"
    rounded_preds = torch.round(preds_to_evaluate)
    
    # Tính độ chính xác: tỷ lệ dự đoán làm tròn bằng với nhãn thực
    # Thêm điều kiện để tránh chia cho 0 nếu không có mẫu nào liên quan
    if len(labels_to_evaluate) > 0:
        sentiment_score = (rounded_preds == labels_to_evaluate).float().mean().item()
    else:
        sentiment_score = 0.0 # Hoặc 1.0 tùy vào định nghĩa, 0.0 an toàn hơn

    # --- 3. Tính Overall Score ---
    overall_score = 0.7 * micro_f1 + 0.3 * sentiment_score

    return micro_f1, sentiment_score, overall_score

def train_and_evaluate(model, train_loader, val_loader, epochs=20, learning_rate=4e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_SE = nn.SmoothL1Loss(beta=0.5, reduction='none') 
    
    loss_CL = nn.BCEWithLogitsLoss() 
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    best_overall_score = -1.0 
    
    scaler = GradScaler()
    print("Bắt đầu huấn luyện...")
    
    for epoch in range(epochs):
        model.train()
        train_total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            texts = batch["text"]
            relevance_labels = batch["relevance_label"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type):
                outputs = model(texts) 
                
                # Giả định model trả về logits cho classification
                relevance_preds = outputs['relevance_logits'] 
                sentiment_preds = outputs['sentiment_preds']
                
                classification_loss = loss_CL(relevance_preds, relevance_labels)
                
                raw_sentiment_loss = loss_SE(sentiment_preds, sentiment_labels)
                
                # 2. Tạo mask từ nhãn relevance
                mask = relevance_labels
                
                # 3. Nhân loss thô với mask
                masked_sentiment_loss = raw_sentiment_loss * mask
                
                # 4. Tính trung bình loss CHỈ trên các phần tử có mask = 1
                sentiment_loss = masked_sentiment_loss.sum() / (mask.sum() + 1e-8)
                # ---------------------------------
                
                alpha = 0.25
                total_loss = alpha * classification_loss + (1 - alpha) * sentiment_loss
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_total_loss += total_loss.item()
            progress_bar.set_postfix({'train_loss': train_total_loss / (progress_bar.n + 1)})

        avg_train_loss = train_total_loss / len(train_loader)

        # Giai đoạn Đánh giá
        micro_f1, sentiment_score, overall_score = evaluate(model, val_loader, device)
        
        # Sửa lỗi: Cập nhật và in ra các metrics mới
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Micro-F1: {micro_f1:.4f} | Sentiment Score: {sentiment_score:.4f} | "
              f"Overall Score: {overall_score:.4f}")

        # Sửa lỗi: Logic lưu model đúng
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            torch.save(model.state_dict(), 'trained_model_final1.pt')
            print(f"*** New best model saved! Overall Score: {best_overall_score:.4f} ***")
        
        print("-" * 80)
        
    print("Hoàn tất huấn luyện.")
    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPb(freeze_encoder=False)
    model.to(device)

    try:
        state_dict = torch.load('trained_model_final1.pt', map_location=device)
        model.load_state_dict(state_dict)
        print("Đã load model từ checkpoint thành công.")
    except FileNotFoundError:
        print("Không tìm thấy file checkpoint, bắt đầu training từ đầu.")

    trained_model = train_and_evaluate(model, train_loader, val_loader)
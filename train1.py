import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from process_data import MyData
from PhObert import MLPb
import csv
import pandas as pd
from torch.utils.data import Dataset

class Mytest(Dataset):
    def __init__(self, csv_path: str):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.texts = str(df["review"])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx]}

train_dataset = MyData('train-problem.csv')
train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

def train1(model, dataloader, epochs=1, save_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.SmoothL1Loss(beta=0.5)  

    optimizer = Adam(model.parameters(), lr=2e-5, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        sum_mae = 0.0
        n_samples = 0
        correct = 0
        total_elems = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            texts = batch["text"]
            labels = batch["label"].to(device)          

            optimizer.zero_grad()
            preds = model(texts)                              

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                mae = torch.abs(preds - labels).mean().item()
                sum_mae += mae * labels.size(0)
                n_samples += labels.size(0)
                correct += (torch.clamp(preds, 0, 5) == labels).sum().item()
                total_elems += labels.numel()
        acc = correct / total_elems
        avg_loss = total_loss / len(dataloader)
        avg_mae = sum_mae / max(1, n_samples)
        print(f"Epoch {epoch+1}/{epochs} | LOSS: {avg_loss:.4f} | MAE: {avg_mae:.4f} | ACC: {acc:.4f}")
        
        last_ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                # bạn có thể ghi thêm các hyperparams vào đây
                "lr": 2e-5,
                "weight_decay": 0.01,
                "loss": "SmoothL1Loss(beta=0.5)",
                "arch": "MLPb(PhoBERT + MLP)"
            }
        }
        torch.save(last_ckpt, f"{save_dir}/last.pt")
        
    print("Done.")

def test_save(model, dataloader, out_csv="submission.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    rows = []
    stt = 1

    with torch.no_grad():
        for batch in dataloader:
            texts = batch['text']
            outputs = model(texts)
            pred_vals = torch.round(outputs).numpy().tolist()

            for p in pred_vals:
                rows.append({"stt": stt, "prediction": p})
                stt += 1

    if not rows:
        print("Không có dữ liệu để ghi CSV.")
        return

    # Ghi CSV với cột đầu tiên là 'stt'
    fieldnames = ["stt", "prediction"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã lưu kết quả vào: {out_csv}")


if __name__ == "__main__":
    model = MLPb(freeze_encoder=True)
    train1(model, train_data, epochs=5, save_dir="checkpoints")
    
''' 
    test_dataset = Mytest('gt_reviews.csv')
    test_data = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    test_save(model,test_data)    
'''
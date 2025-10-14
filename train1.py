import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from process_data import MyData
from PhObert import MLPb

train_dataset = MyData('train-problem.csv')
train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True)

def train1(model, dataloader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.SmoothL1Loss(beta=0.5)  # hoặc nn.SmoothL1Loss(beta=0.5)

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
                correct += (torch.round(preds) == torch.round(labels)).sum().item()
                total_elems += labels.numel()
        acc = correct / total_elems
        avg_loss = total_loss / len(dataloader)
        avg_mae = sum_mae / max(1, n_samples)
        print(f"Epoch {epoch+1}/{epochs} | MSE: {avg_loss:.4f} | MAE: {avg_mae:.4f} | ACC: {acc:.4f}")

    print("Done.")

if __name__ == "__main__":
    model = MLPb(freeze_encoder=True)  # num_classes=6
    train1(model, train_data)

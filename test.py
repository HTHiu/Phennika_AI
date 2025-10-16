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
import os
from PhObert import MLPb

LABEL_COLS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]
class TestDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.texts = dataframe["review"].astype(str).tolist()

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {"text": self.texts[idx]}

# Hàm inference
def predict(model, dataloader, device):
    model.eval()
    all_relevance_preds = []
    all_sentiment_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            texts = batch["text"]
            outputs = model(texts) 

            all_relevance_preds.append(outputs['relevance_preds'].detach().cpu())
            all_sentiment_preds.append(outputs['sentiment_preds'].detach().cpu())

    all_relevance_preds = torch.cat(all_relevance_preds, dim=0)
    all_sentiment_preds = torch.cat(all_sentiment_preds, dim=0)

    binary_relevance = (all_relevance_preds > 0.5).float()     
    rounded_sentiment = torch.clamp(all_sentiment_preds, 0, 5)           
    rounded_sentiment = torch.round(rounded_sentiment)                   


    final_predictions = binary_relevance * rounded_sentiment    

    return final_predictions.numpy() 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPb(freeze_encoder=False).to(device)

    model_path = "trained_model_final0.pt"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded trained model successfully.")
    else:
        raise FileNotFoundError("trained_model_final0.pt not found.")

    test_df = pd.read_csv("/kaggle/input/123456/gt_reviews.csv")
    if 'review' not in test_df.columns:
        raise ValueError("Test file must have 'review' column.")
    test_df['review'] = test_df['review'].fillna('')

    test_dataset = TestDataset(test_df)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, pin_memory=True
    )

    predictions = predict(model, test_loader, device) 
    output_df = pd.DataFrame(predictions, columns=LABEL_COLS)
    output_df.insert(0, 'stt', range(1, len(output_df) + 1))
    output_df.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

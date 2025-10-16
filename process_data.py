import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

LABEL_COLS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]

class MyData(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.texts = dataframe["Review"].astype(str).tolist()
        y = dataframe[LABEL_COLS].values    
        df_relevance = (y > 0).astype(float)
        self.relevance_labels = torch.tensor(df_relevance, dtype=torch.float)
        self.sentiment_labels = torch.tensor(y, dtype=torch.float)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {"text": self.texts[idx], "relevance_label": self.relevance_labels[idx], "sentiment_label": self.sentiment_labels[idx]}



import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer

LABEL_COLS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]

class MyData(Dataset):
    def __init__(self, csv_path: str):
        super().__init__()
        df = pd.read_csv(csv_path)

        self.texts = str(df["Review"])

        y = df[LABEL_COLS].values
        self.labels = torch.tensor(y, dtype=torch.float)   # [N]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

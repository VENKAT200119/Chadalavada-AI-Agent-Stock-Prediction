import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict

# PyTorch imports for CNN-Attention-LSTM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# === XGB.1 Data Ingestion & Windowing ===
# XGB.1.1 Fetch OHLCV for target tickers via yfinance (#604)
def fetch_ohlcv(tickers: List[str], start: str, end: str, interval: str = '1d') -> pd.DataFrame:
    """
    Fetch OHLCV data for given tickers from yfinance.
    Returns a DataFrame with a MultiIndex [Ticker, Date].
    """
    data = yf.download(tickers, start=start, end=end, interval=interval, group_by='ticker', auto_adjust=False)
    df_list = []
    for t in tickers:
        df_t = data[t].copy()
        df_t['Ticker'] = t
        df_list.append(df_t)
    df = pd.concat(df_list)
    df.reset_index(inplace=True)
    df.set_index(['Ticker', 'Date'], inplace=True)
    return df

# XGB.1.2 Impute or drop missing data; align timestamps (#605)
def preprocess_data(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Impute or drop missing data. 
    method: 'ffill', 'bfill', or 'drop'
    """
    if method in ['ffill', 'bfill']:
        df = df.groupby(level=0).apply(lambda x: x.fillna(method=method))
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError("method must be 'ffill', 'bfill', or 'drop'")
    df = df.dropna()
    return df

# XGB.1.3 Normalize features (e.g., z-score) (#606)
def normalize_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Z-score normalization for each ticker separately over the specified feature columns.
    """
    scaler = StandardScaler()
    df_norm = df.copy()
    for t in df.index.get_level_values(0).unique():
        idx = df.index.get_level_values(0) == t
        df_norm.loc[idx, features] = scaler.fit_transform(df.loc[idx, features])
    return df_norm

# XGB.1.4 Slice into overlapping windows & create train/val/test splits (#607)
def slice_windows(
    df: pd.DataFrame,
    features: List[str],
    window_size: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice DataFrame into overlapping windows of length `window_size`.
    Returns:
      X: numpy array of shape (num_windows, window_size, num_features)
      y: numpy array of next-day returns (num_windows,)
    """
    X, y = [], []
    for t in df.index.get_level_values(0).unique():
        df_t = df.loc[t]
        values = df_t[features].values
        closes = df_t['Close'].values
        for start in range(0, len(df_t) - window_size, stride):
            end = start + window_size
            X.append(values[start:end])
            ret = (closes[end] - closes[end - 1]) / closes[end - 1]
            y.append(ret)
    return np.array(X), np.array(y)

def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    shuffle: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split arrays X and y into train/val/test sets according to the given ratios.
    """
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_idx = idx[: n - test_size - val_size]
    val_idx = idx[n - test_size - val_size : n - test_size]
    test_idx = idx[n - test_size :]
    return {
        'train': (X[train_idx], y[train_idx]),
        'val': (X[val_idx], y[val_idx]),
        'test': (X[test_idx], y[test_idx])
    }

# === PyTorch Dataset ===
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === XGB.2 CNN-Attention-LSTM Pretraining ===

# XGB.2.1 Implement 1D-CNN encoder (#644)
# XGB.2.2 Add self-attention layer over time steps (#645)
# XGB.2.3 Stack an LSTM decoder with regression head (#646)
class CNNAttentionLSTM(nn.Module):
    def __init__(self, num_features: int, cnn_channels: int = 32,
                 lstm_hidden: int = 64, attn_heads: int = 4):
        super().__init__()
        # 1D CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Self-attention
        self.attn = nn.MultiheadAttention(embed_dim=cnn_channels,
                                          num_heads=attn_heads,
                                          batch_first=True)
        # LSTM decoder
        self.lstm = nn.LSTM(input_size=cnn_channels,
                            hidden_size=lstm_hidden,
                            batch_first=True)
        # Regression head
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: [batch, time, features]
        x = x.transpose(1, 2)               # -> [batch, features, time]
        x = self.cnn(x)                    # -> [batch, chan, time]
        x = x.transpose(1, 2)              # -> [batch, time, chan]
        x, _ = self.attn(x, x, x)          # -> [batch, time, chan]
        out, (h, _) = self.lstm(x)         # h: [1, batch, hidden]
        emb = h[-1]                        # -> [batch, hidden]
        return self.head(emb), emb         # (pred, embedding)

# XGB.2.4 Write training loop with loss, optimizer, checkpointing, early stopping (#647)
# XGB.2.5 Log metrics and save best model (#648)
def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 50,
    patience: int = 5,
    ckpt_path: str = 'best_model.pt'
) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    wait = 0
    model.to(device)

    for epoch in range(1, epochs+1):
        # train
        model.train()
        tr_losses = []
        for Xb, yb in dataloaders['train']:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in dataloaders['val']:
                Xb, yb = Xb.to(device), yb.to(device)
                pred, _ = model(Xb)
                val_losses.append(criterion(pred, yb).item())
        tr_avg = np.mean(tr_losses)
        val_avg = np.mean(val_losses)
        print(f"Epoch {epoch:02d} | Train {tr_avg:.4f} | Val {val_avg:.4f}")
        if val_avg < best_val:
            best_val = val_avg
            torch.save(model.state_dict(), ckpt_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(ckpt_path))
    return model

# === XGB.3 Embedding Extraction ===

# XGB.3.1 Freeze the trained CNN+Attention+LSTM backbone (#649)
def freeze_backbone(model: nn.Module):
    for p in model.cnn.parameters(): p.requires_grad = False
    for p in model.attn.parameters(): p.requires_grad = False
    for p in model.lstm.parameters(): p.requires_grad = False

# XGB.3.2 & XGB.3.3 Forward windows, extract embeddings, save to disk (#650, #651)
def extract_and_save_embeddings(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    output_file: str = 'embeddings_labels.csv'
):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    embs, labs = [], []
    model.to(device).eval()
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            _, e = model(Xb)
            embs.append(e.cpu().numpy())
            labs.append(yb.numpy())
    embs = np.vstack(embs)
    labs = np.concatenate(labs)
    cols = [f"emb_{i}" for i in range(embs.shape[1])]
    df_out = pd.DataFrame(embs, columns=cols)
    df_out['label'] = labs
    df_out.to_csv(output_file, index=False)
    print(f"Saved embeddings -> {output_file}")

# === Example Usage ===
if __name__ == '__main__':
    # Prompt the user to enter a stock ticker symbol
    ticker_input = input("Enter a stock ticker symbol (e.g., AAPL): ").strip().upper()
    if not ticker_input:
        print("No ticker provided. Exiting.")
        exit(1)

    # Prompt for start and end dates
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    try:
        # Validate basic format
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
    except Exception:
        print("Invalid date format. Please use YYYY-MM-DD. Exiting.")
        exit(1)

    tickers = [ticker_input]
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    window_size = 20

    # 1. Fetch OHLCV data
    df = fetch_ohlcv(tickers, start=start_date, end=end_date)

    # 2. Impute missing values
    df = preprocess_data(df, method='ffill')

    # 3. Normalize features
    df = normalize_features(df, features)

    # 4. Slice into windows and generate labels
    X, y = slice_windows(df, features, window_size)

    # 5. Split into train/val/test
    splits = train_val_test_split(X, y, val_ratio=0.2, test_ratio=0.1, shuffle=True)

    print("\nShapes:")
    print("  Train X:", splits['train'][0].shape, " Train y:", splits['train'][1].shape)
    print("  Val   X:", splits['val'][0].shape,   " Val   y:", splits['val'][1].shape)
    print("  Test  X:", splits['test'][0].shape,  " Test  y:", splits['test'][1].shape)

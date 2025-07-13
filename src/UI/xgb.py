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

# XGBoost & Scikit-Learn imports for fine-tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# === XGB.1 Data Ingestion & Windowing ===
def fetch_ohlcv(tickers: List[str], start: str, end: str,
                interval: str = '1d') -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end,
                       interval=interval, group_by='ticker',
                       auto_adjust=False)
    df_list = []
    for t in tickers:
        df_t = data[t].copy()
        df_t['Ticker'] = t
        df_list.append(df_t)
    df = pd.concat(df_list)
    df.reset_index(inplace=True)
    df.set_index(['Ticker', 'Date'], inplace=True)
    return df

def preprocess_data(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    if method in ['ffill', 'bfill']:
        df = df.groupby(level=0).apply(lambda x: x.fillna(method=method))
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError("method must be 'ffill', 'bfill', or 'drop'")
    return df.dropna()

def normalize_features(df: pd.DataFrame,
                       features: List[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    df_norm = df.copy()
    for t in df_norm.index.get_level_values(0).unique():
        idx = df_norm.index.get_level_values(0) == t
        df_norm.loc[idx, features] = scaler.fit_transform(
            df_norm.loc[idx, features])
    return df_norm

def slice_windows(df: pd.DataFrame, features: List[str],
                  window_size: int, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for t in df.index.get_level_values(0).unique():
        df_t = df.loc[t]
        vals = df_t[features].values
        closes = df_t['Close'].values
        for i in range(0, len(df_t) - window_size, stride):
            X.append(vals[i:i+window_size])
            ret = (closes[i+window_size] - closes[i+window_size-1]) / closes[i+window_size-1]
            y.append(ret)
    return np.array(X), np.array(y)

def train_val_test_split(
    X: np.ndarray, y: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    shuffle: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    t_size = int(n * test_ratio)
    v_size = int(n * val_ratio)
    return {
        'train': (X[idx[:-t_size-v_size]], y[idx[:-t_size-v_size]]),
        'val':   (X[idx[-t_size-v_size:-t_size]], y[idx[-t_size-v_size:-t_size]]),
        'test':  (X[idx[-t_size:]], y[idx[-t_size:]])
    }

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === XGB.2 CNN-Attention-LSTM Pretraining ===
class CNNAttentionLSTM(nn.Module):
    def __init__(self, num_features: int,
                 cnn_channels: int = 32,
                 lstm_hidden: int = 64,
                 attn_heads: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(embed_dim=cnn_channels,
                                          num_heads=attn_heads,
                                          batch_first=True)
        self.lstm = nn.LSTM(input_size=cnn_channels,
                            hidden_size=lstm_hidden,
                            batch_first=True)
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        out, (h, _) = self.lstm(x)
        emb = h[-1]
        return self.head(emb), emb

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
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in dataloaders['val']:
                Xb, yb = Xb.to(device), yb.to(device)
                pred, _ = model(Xb)
                val_losses.append(criterion(pred, yb).item())
        tr_avg, val_avg = np.mean(tr_losses), np.mean(val_losses)
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
def freeze_backbone(model: nn.Module):
    for p in model.cnn.parameters():  p.requires_grad = False
    for p in model.attn.parameters(): p.requires_grad = False
    for p in model.lstm.parameters(): p.requires_grad = False

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

# === XGB.4 XGBoost Fine-tuning ===
# XGB.4.1 Load embeddings + labels into a pandas DataFrame (#652)
def load_embeddings_labels(path: str = 'embeddings_labels.csv') -> pd.DataFrame:
    return pd.read_csv(path)

# XGB.4.2 Perform train/validation split (#653)
def split_embeddings(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return train_test_split(X, y, test_size=val_ratio,
                            random_state=random_state)

# XGB.4.3 Grid-search n_estimators, max_depth, learning_rate (#654)
def tune_xgboost(
    X_train: np.ndarray, y_train: np.ndarray
) -> GridSearchCV:
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    xgb = XGBRegressor(objective='reg:squarederror', verbosity=0)
    grid = GridSearchCV(
        xgb, param_grid, cv=5,
        scoring='neg_mean_squared_error',
        verbose=2, n_jobs=-1,
        error_score='raise'
    )
    # Note: no early_stopping_rounds or eval_set here
    grid.fit(X_train, y_train)
    return grid

# XGB.4.4 Train best estimator; evaluate and save the model (#655)
def train_and_save_xgb(
    grid: GridSearchCV,
    X_val: np.ndarray, y_val: np.ndarray,
    output_path: str = 'xgb_model.joblib'
):
    best = grid.best_estimator_
    preds = best.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    print(f"XGB Validation MSE: {mse:.4f}, R2: {r2:.4f}")
    joblib.dump(best, output_path)
    print(f"Saved XGBoost model -> {output_path}")

# === Main Script ===
if __name__ == '__main__':
    ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    start = input("Start date (YYYY-MM-DD): ").strip()
    end   = input("End date   (YYYY-MM-DD): ").strip()
    try:
        pd.to_datetime(start); pd.to_datetime(end)
    except:
        print("Bad date format."); exit(1)

    features = ['Open','High','Low','Close','Volume']
    win_size = 20

    # XGB.1 pipeline
    df = fetch_ohlcv([ticker], start, end)
    df = preprocess_data(df)
    df = normalize_features(df, features)
    X, y = slice_windows(df, features, win_size)
    splits = train_val_test_split(X, y, val_ratio=0.2,
                                  test_ratio=0.1, shuffle=True)

    batch = 64
    dl = {k: DataLoader(WindowDataset(*splits[k]),
                        batch_size=batch,
                        shuffle=(k=='train'))
          for k in splits}

    # XGB.2 pretraining
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNAttentionLSTM(num_features=len(features))
    model = train_model(model, dl, dev)

    # XGB.3 embedding extraction
    freeze_backbone(model)
    X_all = np.vstack([splits[k][0] for k in splits])
    y_all = np.concatenate([splits[k][1] for k in splits])
    full_ds = WindowDataset(X_all, y_all)
    extract_and_save_embeddings(model, full_ds, dev)

    # XGB.4 fine-tuning
    df_emb = load_embeddings_labels('embeddings_labels.csv')
    X_train, X_val, y_train, y_val = split_embeddings(df_emb)
    grid = tune_xgboost(X_train, y_train)
    train_and_save_xgb(grid, X_val, y_val)

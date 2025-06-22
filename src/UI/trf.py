import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# #########################
# 1. TRF.1: DataAgent
# #########################
class DataAgent:
    """
    Handles data ingestion and preprocessing for OHLCV data.
    """
    def __init__(self, ticker, start_date, end_date, window_size=20):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.scaler = StandardScaler()

    def fetch_ohlcv(self):
        """Fetch Close prices for the given ticker and date range."""
        df = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        # Keep only the 'Close' column
        df = df[['Close']].dropna()
        return df

    def impute_and_align(self, df):
        """Fill missing data and ensure datetime index."""
        df = df.ffill().bfill()
        df.index = pd.to_datetime(df.index)
        return df

    def normalize(self, df):
        """Apply z-score normalization to the Close price."""
        values = df.values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)
        return pd.DataFrame(scaled, index=df.index, columns=df.columns)

    def create_windows(self, df, test_size=0.2, val_size=0.1):
        """
        Build 20-day windows and labels (1 if next-day price up, else 0),
        then split into train/val/test.
        """
        X, y = [], []
        for i in range(len(df) - self.window_size - 1):
            window = df.iloc[i:i+self.window_size].values
            # label based on next-day movement
            label = 1 if df.iloc[i+self.window_size+1, 0] > df.iloc[i+self.window_size, 0] else 0
            X.append(window)
            y.append(label)
        X = np.stack(X)
        y = np.array(y)
        # train + (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size+val_size, shuffle=False
        )
        # split val and test
        val_rel = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_rel, shuffle=False
        )
        return X_train, X_val, X_test, y_train, y_val, y_test


# #########################
# 2. TRF.2: TransformerAgent
# #########################
class TransformerAgent(nn.Module):
    """
    Transformer encoder + classifier for windowed data.
    """
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # project input to model dimension
        self.input_proj = nn.Linear(feature_dim, d_model)
        # transformer encoder layers
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        # classification head
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        # x: [batch, window, feature]
        # project and reorder for transformer: [window, batch, d_model]
        x = self.input_proj(x)  # [batch, window, d_model]
        x = x.permute(1, 0, 2)
        # encode
        out = self.encoder(x)   # [window, batch, d_model]
        # mean-pool over time
        pooled = out.mean(dim=0)  # [batch, d_model]
        # classify
        logits = self.classifier(pooled)
        return logits

    def train_model(self, train_loader, val_loader=None,
                    epochs=20, lr=1e-4, patience=3, ckpt_path='transformer.pt'):
        """Training loop with optional early stopping and checkpoint."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        best_val = float('inf')
        wait = 0
        for epoch in range(1, epochs+1):
            self.train()
            total = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = self(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"Epoch {epoch}, train loss: {total/len(train_loader):.4f}")
            if val_loader:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        val_loss += loss_fn(self(xb), yb).item()
                val_loss /= len(val_loader)
                print(f"Epoch {epoch}, val loss: {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    wait = 0
                    torch.save(self.state_dict(), ckpt_path)
                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping.")
                        break
        # load best model
        self.load_state_dict(torch.load(ckpt_path))


# #########################
# 3. TRF.3: EmbeddingAgent
# #########################
class EmbeddingAgent:
    """
    Extracts embeddings from a trained TransformerAgent.
    """
    def __init__(self, model):
        self.model = model.eval()

    def extract(self, loader, output_csv='embeddings.csv'):
        """Generate embeddings and save with labels."""
        import csv
        device = next(self.model.parameters()).device
        rows = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                # get logits or embeddings before classifier
                # here using classifier logits as embedding
                emb = self.model(xb).cpu().numpy()
                for vec, label in zip(emb, yb.numpy()):
                    rows.append(list(vec) + [int(label)])
        # write CSV
        cols = [f'emb_{i}' for i in range(len(rows[0])-1)] + ['label']
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        print(f"Embeddings saved to {output_csv}")

    def validate(self, csv_path, emb_dim):
        """Check CSV has emb_dim cols + label."""
        df = pd.read_csv(csv_path)
        assert df.shape[1] == emb_dim+1, f"Expected {emb_dim+1} cols, got {df.shape[1]}"
        assert set(df['label'].unique()).issubset({0,1}), "Labels must be 0 or 1"
        print("Embedding file validated.")


# #########################
# Main execution for TRF.1 - TRF.3
# #########################
if __name__ == '__main__':
    # Prompt user for input parameters
    ticker = input("Enter stock ticker (e.g. AAPL): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    # 1. Data ingestion & windowing
    agent = DataAgent(ticker, start_date, end_date)
    df = agent.fetch_ohlcv()
    df = agent.impute_and_align(df)
    df_norm = agent.normalize(df)
    X_train, X_val, X_test, y_train, y_val, y_test = agent.create_windows(df_norm)

    # Wrap data in DataLoader
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    ), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    ), batch_size=32)
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    ), batch_size=32)

    # 2. Transformer model training
    feature_dim = X_train.shape[2]
    transformer = TransformerAgent(feature_dim)
    transformer.train_model(train_loader, val_loader)

    # 3. Embedding extraction
    embedder = EmbeddingAgent(transformer)
    embedder.extract(train_loader, output_csv='embeddings.csv')
    embedder.validate('embeddings.csv', emb_dim=2)  # 2 logits used as embedding size

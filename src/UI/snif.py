import os
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# SNIF.1 Return Data Ingestion
# -----------------------------
class ReturnFetchAgent:
    def __init__(self, source_client=None):
        """
        SNIF.1.1–1.2: Ingest OHLCV and compute/clean returns matrix.
        :param source_client: client with .download(); defaults to yfinance
        """
        self.client = source_client or yf

    def fetch_ohlcv(self, tickers, start, end):
        """
        SNIF.1.1: Fetch OHLCV for tickers between start/end.
        Returns MultiIndex DataFrame (Ticker, [Open,High,Low,Close,Volume]).
        """
        return self.client.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )

    def compute_returns(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        SNIF.1.1: Extract 'Close' prices, compute daily pct_change, drop all-NaN rows.
        Returns DataFrame of shape (days-1)×(n_tickers).
        """
        if isinstance(ohlcv.columns, pd.MultiIndex):
            close = ohlcv.xs("Close", axis=1, level=1)
        else:
            close = ohlcv["Close"]
        return close.pct_change().dropna(how="all")

    def clean_and_align(self, returns: pd.DataFrame, max_nan_pct: float = 0.5) -> pd.DataFrame:
        """
        SNIF.1.2: Drop dates with > max_nan_pct missing data, then forward/backfill gaps.
        """
        thresh = int((1 - max_nan_pct) * returns.shape[1])
        cleaned = returns.dropna(axis=0, thresh=thresh)
        return cleaned.fillna(method="ffill").fillna(method="bfill")


# -----------------------------
# SNIF.2 Autoencoder Training
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # SNIF.2.3: Return latent embeddings
        return self.encoder(x)




# ----------------------------------
# Main SNIF Pipeline Invocation
# ----------------------------------
if __name__ == "__main__":
    # Interactive CLI for SNIF.1
    tickers_input = input("Enter tickers (comma-separated, e.g. AAPL,MSFT,GOOG): ")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    # SNIF.1
    fetcher = ReturnFetchAgent()
    ohlcv = fetcher.fetch_ohlcv(tickers, start_date, end_date)
    print("\nFetched OHLCV (first 5 rows):")
    print(ohlcv.head())

    returns = fetcher.compute_returns(ohlcv)
    print("\nComputed returns (first 5 rows):")
    print(returns.head())

    cleaned = fetcher.clean_and_align(returns)
    print("\nCleaned returns (first 5 rows):")
    print(cleaned.head())

    # SNIF.2
    returns_tensor = torch.tensor(cleaned.values, dtype=torch.float32)
    ae_agent = AutoencoderAgent(input_dim=returns_tensor.shape[1], latent_dim=16, epochs=30)
    ae_agent.train(returns_tensor, val_split=0.2, checkpoint_dir="checkpoints/autoencoder")
    best_encoder = "checkpoints/autoencoder/best_encoder.pth"
    ae_agent.load_encoder(best_encoder)

    embeddings = ae_agent.extract_embeddings(returns_tensor).numpy()
    dates = cleaned.index


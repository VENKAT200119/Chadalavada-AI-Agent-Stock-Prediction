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


class AutoencoderAgent:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 lr: float = 1e-3, batch_size: int = 64, epochs: int = 50,
                 device: str = None):
        """
        SNIF.2.1–2.3: Build, train autoencoder, save encoder weights.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder(input_dim, latent_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs

    def _prepare_dataloaders(self, tensor: torch.Tensor, val_split: float = 0.2):
        dataset = TensorDataset(tensor)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def train(self, returns_array: torch.Tensor,
              val_split: float = 0.2, checkpoint_dir: str = "checkpoints/autoencoder"):
        returns_array = returns_array.to(self.device)
        train_loader, val_loader = self._prepare_dataloaders(returns_array, val_split)
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_loss = float('inf')
        best_path = os.path.join(checkpoint_dir, "best_encoder.pth")

        for epoch in range(1, self.epochs + 1):
            # Training
            self.model.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.criterion(recon, batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch.size(0)
            train_loss /= len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(self.device)
                    val_loss += self.criterion(self.model(batch), batch).item() * batch.size(0)
            val_loss /= len(val_loader.dataset)

            print(f"Epoch {epoch}/{self.epochs}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.encoder.state_dict(), best_path)

        print(f"Training complete. Best Val Loss: {best_loss:.6f}. Saved encoder to {best_path}")

    def load_encoder(self, path: str):
        # SNIF.2.3: Load saved encoder
        self.model.encoder.load_state_dict(torch.load(path, map_location=self.device))
        self.model.encoder.to(self.device).eval()

    def extract_embeddings(self, returns_array: torch.Tensor) -> torch.Tensor:
        # SNIF.2.3: Extract embeddings
        self.model.eval()
        with torch.no_grad():
            return self.model.encode(returns_array.to(self.device)).cpu()

    def save_encoder(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.encoder.state_dict(), path)

# -----------------------------
# SNIF.3 Topology Inference
# -----------------------------
class TopologyAgent:
    def __init__(self, threshold: float = 0.7, output_dir: str = "checkpoints/topology"):
        """
        SNIF.3.1–3.3: Compute/sparsify similarities, persist adjacency.
        """
        self.threshold = threshold
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        # SNIF.3.1: Compute cosine similarities
        return cosine_similarity(embeddings)

    def sparsify(self, sim_matrix: np.ndarray) -> np.ndarray:
        # SNIF.3.2: Threshold to binary adjacency, zero self-links
        adj = (sim_matrix >= self.threshold).astype(float)
        np.fill_diagonal(adj, 0)
        return adj

    def persist(self, adjacency: np.ndarray, dates: pd.DatetimeIndex):
        # SNIF.3.3: Persist adjacency CSVs
        pd.DataFrame(adjacency, index=dates, columns=dates).to_csv(
            f"{self.output_dir}/adjacency_full.csv"
        )
        for i, dt in enumerate(dates):
            pd.DataFrame(
                adjacency[i], index=dates, columns=["edge_weight"]
            ).to_csv(f"{self.output_dir}/adjacency_{dt.strftime('%Y-%m-%d')}.csv")

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

    # SNIF.3
    topo_agent = TopologyAgent(threshold=0.7)
    sim_matrix = topo_agent.compute_similarity(embeddings)
    adjacency = topo_agent.sparsify(sim_matrix)
    topo_agent.persist(adjacency, dates)

    print("\n✅ Full SNIF pipeline completed.")
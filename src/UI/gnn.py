# ============================================
# GNN Pipeline: Price, Graph, and Teacher GNN
# ============================================

import os
from typing import List, Optional, Tuple

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# Get script directory to resolve relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------
# GNN.1 Price & Returns Ingestion (8 ph)
# --------------------------------------------
# GNN.1.1 PriceLoaderAgent: pull OHLC data for all tickers.
# GNN.1.2 Compute daily returns matrix and persist as NumPy/PyTorch tensors.
class PriceLoaderAgent:
    """
    Fetches OHLC data for a universe of tickers and computes daily returns.
    Persists price and returns matrices as NumPy and PyTorch tensors.
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        data_dir: str = "data",
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        # Resolve data_dir relative to script
        self.data_dir = os.path.join(SCRIPT_DIR, data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

        self.ohlc_df: Optional[pd.DataFrame] = None
        self.returns_df: Optional[pd.DataFrame] = None

    def fetch_ohlc_data(self) -> pd.DataFrame:
        raw = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False,
            threads=True,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            adj_close = raw["Adj Close"].copy()
        else:
            adj_close = raw[["Adj Close"]].rename(columns={"Adj Close": self.tickers[0]})

        adj_close = adj_close.reindex(columns=self.tickers)
        adj_close.ffill(inplace=True)
        adj_close.dropna(how="any", inplace=True)

        self.ohlc_df = adj_close
        return adj_close

    def compute_daily_returns(self) -> pd.DataFrame:
        if self.ohlc_df is None:
            raise RuntimeError("OHLC data not yet fetched. Call fetch_ohlc_data() first.")

        prices = self.ohlc_df
        returns = prices.pct_change().iloc[1:].copy()
        returns.dropna(how="any", inplace=True)

        self.returns_df = returns
        return returns

    def persist_tensors(self) -> Tuple[str, str, str, str]:
        if self.ohlc_df is None or self.returns_df is None:
            raise RuntimeError(
                "DataFrames missing. Ensure fetch_ohlc_data() and compute_daily_returns() were called."
            )

        prices_np = self.ohlc_df.values.astype(np.float32)
        returns_np = self.returns_df.values.astype(np.float32)

        prices_pt = torch.from_numpy(prices_np)
        returns_pt = torch.from_numpy(returns_np)

        prices_np_path = os.path.join(self.data_dir, "prices.npy")
        prices_pt_path = os.path.join(self.data_dir, "prices.pt")
        returns_np_path = os.path.join(self.data_dir, "returns.npy")
        returns_pt_path = os.path.join(self.data_dir, "returns.pt")

        np.save(prices_np_path, prices_np)
        torch.save(prices_pt, prices_pt_path)
        np.save(returns_np_path, returns_np)
        torch.save(returns_pt, returns_pt_path)

        return prices_np_path, prices_pt_path, returns_np_path, returns_pt_path


# --------------------------------------------
# GNN.2 Graph Construction (9 ph)
# --------------------------------------------
# GNN.2.1 Implement GraphBuilderAgent that computes N-day rolling Pearson correlations.
# GNN.2.2 Threshold correlations (e.g., |corr| > threshold) to build adjacency matrices.
# GNN.2.3 Save adjacency per date for use in GNNs.
class GraphBuilderAgent:
    """
    Computes N-day rolling Pearson correlations on returns data, thresholds them,
    and saves adjacency matrices (NumPy and PyTorch) per date.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        window_size: int,
        threshold: float,
        output_dir: str = "adjacency",
    ):
        self.returns_df = returns_df.copy()
        self.window_size = window_size
        self.threshold = threshold
        # Resolve output_dir relative to script
        self.output_dir = os.path.join(SCRIPT_DIR, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def build_graphs(self) -> None:
        dates = list(self.returns_df.index)

        for idx in range(self.window_size - 1, len(dates)):
            window_slice = self.returns_df.iloc[idx - self.window_size + 1 : idx + 1]
            corr_mat = window_slice.corr().values.astype(np.float32)

            # Threshold correlations
            adj_np = (np.abs(corr_mat) > self.threshold).astype(np.float32)
            np.fill_diagonal(adj_np, 0.0)

            # Convert to PyTorch tensor
            adj_pt = torch.from_numpy(adj_np)

            date_str = dates[idx].strftime("%Y-%m-%d")
            np_path = os.path.join(self.output_dir, f"adjacency_{date_str}.npy")
            pt_path = os.path.join(self.output_dir, f"adjacency_{date_str}.pt")

            np.save(np_path, adj_np)
            torch.save(adj_pt, pt_path)


# --------------------------------------------
# GNN.3 Teacher GNN Development (9 ph)
# --------------------------------------------
# GNN.3.1 Design a temporal GNN (e.g., GCN+GRU) that takes node features up to t+Δ.
# GNN.3.2 Implement training loop with cross-entropy or MSE loss on next-day labels.
# GNN.3.3 Log train/val accuracy and save best teacher checkpoint.
class TeacherGNN(nn.Module):
    """
    Temporal GNN combining a GCN layer with a GRU for node-level sequence modeling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        gru_hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.gcn = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True)
        self.classifier = nn.Linear(gru_hidden_dim, output_dim)

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        seq_len, N = features.shape
        x = features.unsqueeze(-1)  # (seq_len, N, 1)
        hidden_seq = []
        for t in range(seq_len):
            h = adj @ x[t]           # (N, 1)
            h = self.gcn(h)          # (N, hidden_dim)
            hidden_seq.append(h)

        hidden_seq = torch.stack(hidden_seq, dim=0).permute(1, 0, 2)
        out, _ = self.gru(hidden_seq)  # (N, seq_len, gru_hidden_dim)
        last = out[:, -1, :]           # (N, gru_hidden_dim)
        logits = self.classifier(last) # (N, output_dim)
        return logits


class TeacherTrainAgent:
    """
    Implements training of the Teacher GNN on adjacency+feature sequences.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        adj_dir: str,
        delta: int,
        model_dir: str,
        lr: float = 1e-3,
        epochs: int = 10,
    ):
        self.returns_df = returns_df.copy()
        # Resolve adj_dir relative to script
        self.adj_dir = os.path.join(SCRIPT_DIR, adj_dir)
        self.delta = delta
        self.model_dir = os.path.join(SCRIPT_DIR, model_dir)
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.model_dir, exist_ok=True)

        self.num_nodes = len(returns_df.columns)
        self.model = TeacherGNN(
            input_dim=1, hidden_dim=32, gru_hidden_dim=64, output_dim=2
        ).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.train_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.val_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def prepare_data(self) -> None:
        dates = list(self.returns_df.index)
        features, adjs, labels = [], [], []

        # Build samples from delta to len-1, skip if adjacency file missing
        for idx in range(self.delta, len(dates) - 1):
            seq_start = idx - self.delta
            seq_end = idx + self.delta
            feat_seq = self.returns_df.iloc[seq_start : seq_end + 1].values.astype(np.float32)

            date_str = dates[idx].strftime("%Y-%m-%d")
            adj_path = os.path.join(self.adj_dir, f"adjacency_{date_str}.npy")
            if not os.path.exists(adj_path):
                print(f"Warning: adjacency file missing for {date_str}, skipping sample.")
                continue
            adj_np = np.load(adj_path).astype(np.float32)

            next_ret = self.returns_df.iloc[idx + 1].values
            label = (next_ret > 0).astype(int)

            features.append(feat_seq)
            adjs.append(adj_np)
            labels.append(label)

        if not features:
            raise RuntimeError("No training samples available; check adjacency files and settings.")

        split = int(0.8 * len(features))
        self.train_data = list(zip(features[:split], adjs[:split], labels[:split]))
        self.val_data = list(zip(features[split:], adjs[split:], labels[split:]))

    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            # Training loop
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            for feat_np, adj_np, label_np in self.train_data:
                feat = torch.tensor(feat_np, dtype=torch.float32, device=self.device)
                adj = torch.tensor(adj_np, dtype=torch.float32, device=self.device)
                label = torch.tensor(label_np, dtype=torch.long, device=self.device)

                logits = self.model(feat, adj)
                loss = self.criterion(logits, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)

            train_acc = correct / total

            # Validation loop
            self.model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for feat_np, adj_np, label_np in self.val_data:
                    feat = torch.tensor(feat_np, dtype=torch.float32, device=self.device)
                    adj = torch.tensor(adj_np, dtype=torch.float32, device=self.device)
                    label = torch.tensor(label_np, dtype=torch.long, device=self.device)

                    logits = self.model(feat, adj)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == label).sum().item()
                    val_total += label.size(0)

            val_acc = val_correct / val_total
            print(f"Epoch {epoch}/{self.epochs} - Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            # Save best checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                ckpt_path = os.path.join(self.model_dir, "best_teacher.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"New best model saved to {ckpt_path}")


if __name__ == "__main__":
    # 1) Price & Returns Ingestion
    user_input = input("Enter stock tickers separated by commas (e.g., AAPL,MSFT,GOOG): ")
    universe = [t.strip().upper() for t in user_input.split(",") if t.strip()]
    start, end = "2020-01-01", "2023-12-31"

    loader = PriceLoaderAgent(universe, start, end, data_dir="gnn_data")
    prices_df = loader.fetch_ohlc_data()
    print("Price DataFrame head:\n", prices_df.head())

    returns_df = loader.compute_daily_returns()
    print("\nReturns DataFrame head:\n", returns_df.head())

    loader.persist_tensors()

    # 2) Graph Construction
    window = int(input("Enter rolling window size (e.g., 30): "))
    thresh = float(input("Enter correlation threshold (e.g., 0.7): "))
    graph_builder = GraphBuilderAgent(returns_df, window, thresh, output_dir="adjacency_data")
    graph_builder.build_graphs()
    print("Adjacency matrices saved to 'adjacency_data/'.")

    # 3) Teacher GNN Training
    delta = int(input("Enter future window Δ (e.g., 1): "))
    epochs = int(input("Enter number of training epochs (e.g., 10): "))
    lr = float(input("Enter learning rate (e.g., 0.001): "))

    teacher_agent = TeacherTrainAgent(returns_df, "adjacency_data", delta, model_dir="teacher_model", lr=lr, epochs=epochs)
    teacher_agent.prepare_data()
    teacher_agent.train()

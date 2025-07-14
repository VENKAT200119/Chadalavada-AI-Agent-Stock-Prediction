# ============================================
# GNN Pipeline: Price, Graph, Teacher & Student GNN, Inference
# ============================================

import os
import sys
from typing import List, Optional, Tuple, Dict, Any
import json
import time

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Get script directory to resolve relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------
# GNN.1 Price & Returns Ingestion (8 ph)
# --------------------------------------------
class PriceLoader:
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
class GraphBuilder:
    """
    Computes N-day rolling Pearson correlations on returns data, thresholds them,
    and saves adjacency matrices (NumPy and PyTorch) per date.
    """
    def __init__(
        self,
        returns_df: pd.DataFrame,
        window_size: int,
        threshold: float,
        output_dir: str = "adjacency_data",
    ):
        self.returns_df = returns_df.copy()
        self.window_size = window_size
        self.threshold = threshold
        self.output_dir = os.path.join(SCRIPT_DIR, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def build_graphs(self) -> None:
        dates = list(self.returns_df.index)
        for idx in range(self.window_size - 1, len(dates)):
            window_slice = self.returns_df.iloc[idx - self.window_size + 1 : idx + 1]
            corr_mat = window_slice.corr().values.astype(np.float32)
            adj_np = (np.abs(corr_mat) > self.threshold).astype(np.float32)
            np.fill_diagonal(adj_np, 0.0)
            adj_pt = torch.from_numpy(adj_np)
            date_str = dates[idx].strftime("%Y-%m-%d")
            np_path = os.path.join(self.output_dir, f"adjacency_{date_str}.npy")
            pt_path = os.path.join(self.output_dir, f"adjacency_{date_str}.pt")
            np.save(np_path, adj_np)
            torch.save(adj_pt, pt_path)

# --------------------------------------------
# GNN.3 Teacher GNN Development (9 ph)
# --------------------------------------------
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
            h = adj @ x[t]
            h = self.gcn(h)
            hidden_seq.append(h)
        hidden_seq = torch.stack(hidden_seq, dim=0).permute(1, 0, 2)
        out, _ = self.gru(hidden_seq)
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits

class TeacherTrain:
    """
    Training loop for Teacher GNN on adjacency and return sequences.
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
        self.adj_dir = os.path.join(SCRIPT_DIR, adj_dir)
        self.delta = delta
        self.model_dir = os.path.join(SCRIPT_DIR, model_dir)
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = TeacherGNN(input_dim=1, hidden_dim=32, gru_hidden_dim=64, output_dim=2).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.train_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.val_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def prepare_data(self) -> None:
        dates = list(self.returns_df.index)
        feats, adjs, labels = [], [], []
        for idx in range(self.delta, len(dates) - 1):
            seq_start = idx - self.delta
            seq_end = idx + self.delta
            feat_seq = self.returns_df.iloc[seq_start : seq_end + 1].values.astype(np.float32)
            date_str = dates[idx].strftime("%Y-%m-%d")
            adj_path = os.path.join(self.adj_dir, f"adjacency_{date_str}.npy")
            if not os.path.exists(adj_path):
                continue
            adj_np = np.load(adj_path).astype(np.float32)
            next_ret = self.returns_df.iloc[idx + 1].values
            label = (next_ret > 0).astype(int)
            feats.append(feat_seq)
            adjs.append(adj_np)
            labels.append(label)
        if not feats:
            raise RuntimeError("No training data; check adjacency files.")
        split = int(0.8 * len(feats))
        self.train_data = list(zip(feats[:split], adjs[:split], labels[:split]))
        self.val_data = list(zip(feats[split:], adjs[split:], labels[split:]))

    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            for feat_np, adj_np, label_np in self.train_data:
                feat = torch.tensor(feat_np, device=self.device)
                adj = torch.tensor(adj_np, device=self.device)
                label = torch.tensor(label_np, device=self.device)
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

            self.model.eval()
            val_corr, val_tot = 0, 0
            with torch.no_grad():
                for feat_np, adj_np, label_np in self.val_data:
                    feat = torch.tensor(feat_np, device=self.device)
                    adj = torch.tensor(adj_np, device=self.device)
                    label = torch.tensor(label_np, device=self.device)
                    logits = self.model(feat, adj)
                    preds = logits.argmax(dim=1)
                    val_corr += (preds == label).sum().item()
                    val_tot += label.size(0)
            val_acc = val_corr / val_tot

            print(f"Epoch {epoch}/{self.epochs} - Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                ckpt = os.path.join(self.model_dir, "best_teacher.pth")
                torch.save(self.model.state_dict(), ckpt)
                print(f"Saved best teacher model to {ckpt}")

# --------------------------------------------
# GNN.4 Student GNN & Distillation (7 ph)
# --------------------------------------------
class StudentGNN(TeacherGNN):
    """
    Student model: same as teacher but no future inputs (uses only features up to t).
    """
    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return super().forward(features, adj)

class StudentTrain:
    """
    Train student with distillation loss (KL + classification).
    """
    def __init__(
        self,
        returns_df: pd.DataFrame,
        adj_dir: str,
        delta: int,
        teacher_ckpt: str,
        model_dir: str,
        lr: float = 1e-3,
        epochs: int = 10,
        alpha: float = 0.5,
        temperature: float = 2.0,
    ):
        self.returns_df = returns_df.copy()
        self.adj_dir = os.path.join(SCRIPT_DIR, adj_dir)
        self.delta = delta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = TeacherGNN(1, 32, 64, 2).to(self.device)
        self.teacher.load_state_dict(torch.load(teacher_ckpt, map_location=self.device))
        self.teacher.eval()
        self.model = StudentGNN(1, 32, 64, 2).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.alpha = alpha
        self.T = temperature
        self.epochs = epochs
        self.model_dir = os.path.join(SCRIPT_DIR, model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.train_data, self.val_data = [], []

    def prepare_data(self):
        dates = list(self.returns_df.index)
        feats, adjs, labels = [], [], []
        for idx in range(self.delta, len(dates) - 1):
            seq_start = idx - self.delta
            seq_end = idx + 1  # student sees only up to current
            feat_seq = self.returns_df.iloc[seq_start : seq_end].values.astype(np.float32)
            date_str = dates[idx].strftime("%Y-%m-%d")
            adj_path = os.path.join(self.adj_dir, f"adjacency_{date_str}.npy")
            if not os.path.exists(adj_path):
                continue
            adj_np = np.load(adj_path).astype(np.float32)
            next_ret = self.returns_df.iloc[idx + 1].values
            label = (next_ret > 0).astype(int)
            feats.append(feat_seq)
            adjs.append(adj_np)
            labels.append(label)
        split = int(0.8 * len(feats))
        self.train_data = list(zip(feats[:split], adjs[:split], labels[:split]))
        self.val_data = list(zip(feats[split:], adjs[split:], labels[split:]))

    def train(self):
        best_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            self.model.train(); total_loss = 0.0; correct = 0; total = 0
            for feat_np, adj_np, label_np in self.train_data:
                feat = torch.tensor(feat_np, device=self.device)
                adj = torch.tensor(adj_np, device=self.device)
                label = torch.tensor(label_np, device=self.device)
                s_logits = self.model(feat, adj)
                with torch.no_grad():
                    t_logits = self.teacher(feat, adj)
                loss_ce = self.ce(s_logits, label)
                p_s = F.log_softmax(s_logits / self.T, dim=1)
                p_t = F.softmax(t_logits / self.T, dim=1)
                loss_kl = self.kl(p_s, p_t) * (self.T * self.T)
                loss = self.alpha * loss_kl + (1 - self.alpha) * loss_ce
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                total_loss += loss.item()
                preds = s_logits.argmax(dim=1)
                correct += (preds == label).sum().item(); total += label.size(0)
            train_acc = correct / total

            self.model.eval(); v_corr = 0; v_tot = 0
            with torch.no_grad():
                for feat_np, adj_np, label_np in self.val_data:
                    feat = torch.tensor(feat_np, device=self.device)
                    adj = torch.tensor(adj_np, device=self.device)
                    label = torch.tensor(label_np, device=self.device)
                    logits = self.model(feat, adj)
                    preds = logits.argmax(dim=1)
                    v_corr += (preds == label).sum().item(); v_tot += label.size(0)
            val_acc = v_corr / v_tot

            print(f"Epoch {epoch}/{self.epochs} - Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                path = os.path.join(self.model_dir, "best_student.pth")
                torch.save(self.model.state_dict(), path)
                print(f"Saved best student model to {path}")

# --------------------------------------------
# GNN.5 Inference Wrapper (6 ph)
# --------------------------------------------
class Inference:
    """
    Loads a distilled student model and runs inference on today's data.
    """
    def __init__(self, model_path: str, tickers: List[str], delta: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StudentGNN(1, 32, 64, 2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.tickers = tickers
        self.delta = delta
        # load returns
        self.returns = np.load(os.path.join(SCRIPT_DIR, "data", "returns.npy"))

    def run(self) -> List[Dict[str, Any]]:
        # prepare last sequence
        seq = self.returns[-self.delta:, :]

        # pick only .npy adjacency files
        adj_dir = os.path.join(SCRIPT_DIR, "adjacency_data")
        npy_files = [f for f in os.listdir(adj_dir)
                     if f.startswith("adjacency_") and f.endswith(".npy")]
        if not npy_files:
            raise RuntimeError(f"No .npy files found in {adj_dir}")
        npy_files.sort()
        latest = npy_files[-1]
        adj_np = np.load(os.path.join(adj_dir, latest)).astype(np.float32)

        feat = torch.tensor(seq, device=self.device)
        adj_t = torch.tensor(adj_np, device=self.device)
        logits = self.model(feat, adj_t)
        probs = F.softmax(logits, dim=1)[:, 1]  # probability up
        return [{"stock": ticker, "student_prob": p}
                for ticker, p in zip(self.tickers, probs.tolist())]

if __name__ == "__main__":
    # 1) Price Loading
    universe = input("Enter tickers separated by commas: ").upper().split(",")
    loader = PriceLoader(universe, "2020-01-01", "2023-12-31", data_dir="data")
    prices = loader.fetch_ohlc_data()
    returns = loader.compute_daily_returns()
    loader.persist_tensors()

    # 2) Graphs
    window = int(input("Enter rolling window size: "))
    thresh = float(input("Enter correlation threshold: "))
    gb = GraphBuilder(returns, window, thresh, output_dir="adjacency_data")
    gb.build_graphs()

    # 3) Teacher Train
    delta = int(input("Enter teacher delta: "))
    epochs_t = int(input("Enter teacher epochs: "))
    lr_t = float(input("Enter teacher learning rate: "))
    tch = TeacherTrain(returns, "adjacency_data",
                       delta,
                       model_dir="teacher_model",
                       lr=lr_t,
                       epochs=epochs_t)
    tch.prepare_data()
    tch.train()

    # 4) Student Train & Distillation
    epochs_s = int(input("Enter student epochs: "))
    tch_ckpt = os.path.join(SCRIPT_DIR, "teacher_model", "best_teacher.pth")
    stu = StudentTrain(returns,
                       "adjacency_data",
                       delta,
                       teacher_ckpt=tch_ckpt,
                       model_dir="student_model",
                       lr=lr_t,
                       epochs=epochs_s)
    stu.prepare_data()
    stu.train()

    # 5) Inference
    infer = Inference(os.path.join(SCRIPT_DIR, "student_model", "best_student.pth"),
                      universe,
                      delta)
    start = time.time()
    result = infer.run()
    duration = time.time() - start
    print(json.dumps({"inference_time": duration, "predictions": result}, indent=2))

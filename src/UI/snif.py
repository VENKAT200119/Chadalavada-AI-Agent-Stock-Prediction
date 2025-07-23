import os
import sys
import json
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

# Ensure repo root is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# -----------------------------
# SNIF.1 Return Data Ingestion
# -----------------------------
class ReturnFetcher:
    def __init__(self, source_client=None):
        """SNIF.1.1–1.2: Fetch and clean OHLCV returns."""
        self.client = source_client or yf

    def fetch_ohlcv(self, tickers, start, end):
        """Download OHLCV data."""
        return self.client.download(
            tickers, start=start, end=end,
            group_by='ticker', auto_adjust=False, progress=False
        )

    def compute_returns(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute daily pct_change of Close prices."""
        if isinstance(ohlcv.columns, pd.MultiIndex):
            close = ohlcv.xs('Close', axis=1, level=1)
        else:
            close = ohlcv['Close']
        return close.pct_change().dropna(how='all')

    def clean_and_align(self, returns: pd.DataFrame, max_nan_pct: float = 0.5) -> pd.DataFrame:
        """SNIF.1.2: Drop dates with too many NaNs, then ffill/bfill."""
        thresh = int((1 - max_nan_pct) * returns.shape[1])
        cleaned = returns.dropna(axis=0, thresh=thresh)
        return cleaned.ffill().bfill()

# -----------------------------
# SNIF.2 Autoencoder Training
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(True),
            nn.Linear(128, latent_dim), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(True),
            nn.Linear(128, input_dim), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class AutoencoderTrainer:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 lr: float = 1e-3, batch_size: int = 64,
                 epochs: int = 50, device: str = None):
        """SNIF.2: Build/train autoencoder and save encoder."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Autoencoder(input_dim, latent_dim).to(self.device)
        self.crit = nn.MSELoss()
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs

    def _prepare_loaders(self, data: torch.Tensor, val_split=0.2):
        ds = TensorDataset(data)
        val_size = int(len(ds) * val_split)
        train_ds, val_ds = random_split(ds, [len(ds)-val_size, val_size])
        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=self.batch_size)
        )

    def train(self, data: torch.Tensor,
              checkpoint_dir: str = 'src/UI/checkpoints/autoencoder'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_loss = float('inf')
        best_path = os.path.join(checkpoint_dir, 'best_encoder.pth')
        train_loader, val_loader = self._prepare_loaders(data.to(self.device))
        for ep in range(1, self.epochs+1):
            self.model.train()
            train_loss=0
            for (batch,) in train_loader:
                self.opt.zero_grad()
                recon = self.model(batch.to(self.device))
                loss = self.crit(recon, batch.to(self.device))
                loss.backward(); self.opt.step()
                train_loss += loss.item()*batch.size(0)
            train_loss /= len(train_loader.dataset)
            self.model.eval()
            val_loss=0
            with torch.no_grad():
                for (batch,) in val_loader:
                    val_loss += self.crit(self.model(batch.to(self.device)), batch.to(self.device)).item()*batch.size(0)
            val_loss /= len(val_loader.dataset)
            print(f'Epoch {ep}/{self.epochs}: Train {train_loss:.4f}, Val {val_loss:.4f}')
            if val_loss<best_loss:
                best_loss=val_loss; torch.save(self.model.encoder.state_dict(), best_path)
        print('SNIF.2 complete: encoder at', best_path)

    def load_encoder(self, path: str):
        """Load trained encoder weights."""
        self.model.encoder.load_state_dict(torch.load(path, map_location=self.device))
        self.model.encoder.eval()

    def extract_embeddings(self, data: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad(): return self.model.encode(data.to(self.device)).cpu()

# -----------------------------
# SNIF.3 Topology Inference
# -----------------------------
class TopologyBuilder:
    def __init__(self, threshold=0.7,
                 output_dir='src/UI/checkpoints/topology'):
        os.makedirs(output_dir, exist_ok=True)
        self.threshold=threshold; self.output_dir=output_dir
    def compute_similarity(self, emb: np.ndarray) -> np.ndarray:
        return cosine_similarity(emb)
    def sparsify(self, sim: np.ndarray) -> np.ndarray:
        adj=(sim>=self.threshold).astype(float); np.fill_diagonal(adj,0); return adj
    def persist(self, adj: np.ndarray, dates: pd.DatetimeIndex):
        pd.DataFrame(adj,index=dates,columns=dates).to_csv(os.path.join(self.output_dir,'adj_full.csv'))
        for i,dt in enumerate(dates):
            pd.DataFrame(adj[i],index=dates,columns=['weight']).to_csv(
                os.path.join(self.output_dir,f'adj_{dt.date()}.csv'))

# -----------------------------
# SNIF.4 GCN+LSTM Development
# -----------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)
    def forward(self, x, adj):
        I = torch.eye(adj.size(0), device=adj.device)
        A_hat = adj + I
        deg = A_hat.sum(1)
        D_inv_sqrt = torch.diag(deg.pow(-0.5))
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        return F.relu(self.linear(A_norm @ x))

class GCN_LSTM(nn.Module):
    def __init__(self, feat_dim, gcn_hidden=64, lstm_hidden=64, num_classes=3):
        """
        SNIF.4.1: Two-layer GCN; SNIF.4.2: LSTM; SNIF.4.4: FC head
        """
        super().__init__()
        # Rename to match checkpoint keys
        self.gcn1 = GCNLayer(feat_dim, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden)
        self.lstm = nn.LSTM(gcn_hidden, lstm_hidden, num_layers=2, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, seq: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # seq: [batch=1, seq_len, feat_dim]
        # we treat seq_len as number of time steps or snapshots
        outs = []
        for t in range(seq.size(1)):
            x = seq[:, t, :].squeeze(0)  # [nodes, feat]
            x = self.gcn1(x, adj)
            x = self.gcn2(x, adj)
            outs.append(x.mean(dim=0, keepdim=True))
        # Build LSTM input: [batch=1, seq_len, feat]
        lstm_input = torch.cat(outs, dim=0).unsqueeze(0)
        lstm_out, _ = self.lstm(lstm_input)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

# -----------------------------
# SNIF.5 Inference
# -----------------------------
class InferenceEngine:
    def __init__(self,model_pt,enc_pth,input_dim,latent_dim,thresh=0.7):
        self.m=torch.jit.load(model_pt);self.m.eval()
        self.enc=AutoencoderTrainer(input_dim=input_dim,latent_dim=latent_dim)
        self.enc.load_encoder(enc_pth)
        self.top=TopologyBuilder(thresh)
    def run(self,tickers,start,end):
        rf=ReturnFetcher();oh=rf.fetch_ohlcv(tickers,start,end)
        rt=rf.compute_returns(oh);cl=rf.clean_and_align(rt)
        emb=self.enc.extract_embeddings(torch.tensor(cl.values,dtype=torch.float32)).numpy()
        adj=torch.tensor(self.top.sparsify(self.top.compute_similarity(emb)),dtype=torch.float32)
        # Prepare sequence for GCN+LSTM: [1, num_nodes, feat_dim]
        # Prepare sequence for GCN+LSTM: [seq_len, num_nodes, feat_dim]
        seq = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Inference through TorchScript model
        pr = F.softmax(self.m(seq, adj).squeeze(0), dim=-1)
        return json.dumps([{'stock':s,'snif_prob':pr[i].item()} for i,s in enumerate(tickers)])

if __name__=='__main__':
    # Input tickers and dates separately to avoid unpack errors
    tickers = input('Enter tickers (comma-separated): ').split(',')
    start_date = input('Enter start date (YYYY-MM-DD): ')
    end_date = input('Enter end date (YYYY-MM-DD): ')

    # SNIF.1–3
    rf = ReturnFetcher()
    oh = rf.fetch_ohlcv(tickers, start_date, end_date)
    rt = rf.compute_returns(oh)
    cl = rf.clean_and_align(rt)

    ae = AutoencoderTrainer(input_dim=cl.shape[1], latent_dim=16, epochs=30)
    ae.train(torch.tensor(cl.values, dtype=torch.float32))
    enc_pth = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'checkpoints', 'autoencoder', 'best_encoder.pth')
    )
    ae.load_encoder(enc_pth)

    # SNIF.3
    emb = ae.extract_embeddings(torch.tensor(cl.values, dtype=torch.float32)).numpy()
    tb = TopologyBuilder()
    tb.persist(tb.sparsify(tb.compute_similarity(emb)), cl.index)
    print('SNIF.3 complete.')

    # SNIF.4: Serialize GCN_LSTM using current architecture
    latent_dim = 16
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_dir = os.path.join(root, 'models')
    os.makedirs(model_dir, exist_ok=True)
    pth = os.path.join(model_dir, 'snif_student.pth')
    pt  = os.path.join(model_dir, 'snif_student.pt')
    gcn_model = GCN_LSTM(feat_dim=latent_dim)
    if os.path.exists(pth):
        gcn_model.load_state_dict(torch.load(pth, map_location='cpu'))
    else:
        torch.save(gcn_model.state_dict(), pth)
    ts_model = torch.jit.script(gcn_model)
    ts_model.save(pt)
    print('SNIF.4 complete: scripted GCN_LSTM to', pt)

    # SNIF.5
    engine = InferenceEngine(pt, enc_pth, input_dim=cl.shape[1], latent_dim=latent_dim)
    result = engine.run(tickers, start_date, end_date)
    print('SNIF.5 complete:')
    print('Inference output:', result)

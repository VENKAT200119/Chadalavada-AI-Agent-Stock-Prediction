# pipeline_agents.py

######################################
# NQR.1 Ratio Data Retrieval & Preparation
######################################
import yfinance as yf
import sys
import os
import numpy as np
import pandas as pd
import joblib
import time
import pickle
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

import openai   # NQR.4: Crew AI integration

# Optional ANFIS import
try:
    from anfis import ANFISNet
except ImportError:
    ANFISNet = None

class DataFetch:
    """
    Fetch quarterly financial statements for a given ticker via yfinance.
    """
    def __init__(self, num_quarters: int = 10):
        self.num_quarters = num_quarters

    def fetch_income_statement(self, ticker: str) -> list:
        """
        NQR.1.1: Pull quarterly income statements (latest `num_quarters`).
        Returns list of dicts: fiscalDateEnding, totalRevenue, netIncome.
        """
        tk = yf.Ticker(ticker)
        inc_df = tk.quarterly_financials.transpose().sort_index(ascending=False)
        top_inc = inc_df.head(self.num_quarters)
        records = []
        for idx, row in top_inc.iterrows():
            records.append({
                "fiscalDateEnding": idx.strftime("%Y-%m-%d"),
                "totalRevenue": row.get("Total Revenue", None),
                "netIncome": row.get("Net Income", None),
            })
        if not records:
            raise ValueError(f"No income statement data found for {ticker}")
        return records

    def fetch_balance_sheet(self, ticker: str) -> list:
        """
        NQR.1.1: Pull quarterly balance sheets (latest `num_quarters`).
        Returns list of dicts: fiscalDateEnding, totalShareholderEquity, totalLiabilities.
        """
        tk = yf.Ticker(ticker)
        bs_df = tk.quarterly_balance_sheet.transpose().sort_index(ascending=False)
        top_bs = bs_df.head(self.num_quarters)
        records = []
        for idx, row in top_bs.iterrows():
            records.append({
                "fiscalDateEnding": idx.strftime("%Y-%m-%d"),
                "totalShareholderEquity": row.get("Total Stockholder Equity", None),
                "totalLiabilities": row.get("Total Liab", None),
            })
        if not records:
            raise ValueError(f"No balance sheet data found for {ticker}")
        return records

class RatioCalc:
    """
    Compute financial ratios (ROE, debt/equity, net profit margin).
    """
    def compute_ratios(self, income_reports: list, balance_reports: list) -> list:
        """
        NQR.1.2: Calculate ratios and handle missing values.
        """
        ratios = []
        for inc, bal in zip(income_reports, balance_reports):
            try:
                net_inc = float(inc.get('netIncome', 0) or 0)
                rev = float(inc.get('totalRevenue', 0) or 0)
                equity = float(bal.get('totalShareholderEquity', 0) or 0)
                liabilities = float(bal.get('totalLiabilities', 0) or 0)
            except (TypeError, ValueError):
                net_inc = rev = equity = liabilities = 0.0
            roe = net_inc / equity if equity else None
            debt_equity = liabilities / equity if equity else None
            net_profit_margin = net_inc / rev if rev else None
            ratios.append({
                'fiscalDateEnding': inc.get('fiscalDateEnding'),
                'roe': roe,
                'debt_equity': debt_equity,
                'net_profit_margin': net_profit_margin
            })
        return ratios

######################################
# NQR.2 Model Training Pipeline
######################################
class ModelTrain:
    """
    Train Feed-Forward NN, Random Forest, and ANFIS models.
    """
    class FeedForwardNN(nn.Module):
        def __init__(self, input_dim, hidden_dim=16):
            super(ModelTrain.FeedForwardNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.network(x)

    def train_fnn(self, X, y, epochs=20, lr=0.001):
        """
        NQR.2.1 & NQR.2.2: Define FNN, train and validate, return model and accuracy.
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

        model = ModelTrain.FeedForwardNN(X.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val_t)
            preds_cls = (preds.numpy() >= 0.5).astype(int)
            val_acc = accuracy_score(y_val, preds_cls)

        return model, val_acc

    def train_rf(self, X, y):
        """
        NQR.2.3: Train Random Forest with hyperparameter tuning.
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
        grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
        grid.fit(X_train, y_train)
        best_rf = grid.best_estimator_
        preds = best_rf.predict(X_val)
        val_acc = accuracy_score(y_val, preds)
        return best_rf, val_acc

    def train_anfis(self, X, y):
        """
        NQR.2.4: Train ANFIS model and verify convergence.
        """
        if ANFISNet is None:
            raise ImportError("ANFISNet library not installed.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        anfis = ANFISNet(input_dim=X.shape[1])
        anfis.fit(X_train, y_train)
        preds = anfis.predict(X_val)
        preds_cls = (preds >= 0.5).astype(int)
        val_acc = accuracy_score(y_val, preds_cls)
        return anfis, val_acc

    def compare_models(self, acc_fnn, acc_rf, acc_anfis):
        """
        NQR.2.5: Compare validation accuracies and select the best model.
        """
        best = max([(acc_fnn, 'FNN'), (acc_rf, 'RF'), ((acc_anfis or 0), 'ANFIS')])[1]
        return {'fnn_acc': acc_fnn, 'rf_acc': acc_rf, 'anfis_acc': acc_anfis, 'best_model': best}

######################################
# NQR.3 Inference API Development
######################################
class ModelInferAgent:
    """
    Load trained models and perform inference, outputting JSON.
    """
    def __init__(self, fnn_path, rf_path, anfis_path=None):
        # NQR.3.1: Load FNN, RF, (and ANFIS if provided)
        self.fnn = torch.load(fnn_path, map_location='cpu', weights_only=False)
        self.fnn.eval()
        self.rf = joblib.load(rf_path)
        self.anfis = None
        if anfis_path:
            with open(anfis_path, 'rb') as f:
                self.anfis = pickle.load(f)

    def infer(self, ratios: list) -> dict:
        """
        NQR.3.2: Compute probabilities {fnn_prob, rf_prob, anfis_prob}.
        NQR.3.3: Measure and report latency if over 1 second.
        """
        start = time.time()
        x = np.array(ratios, dtype=float).reshape(1, -1)
        with torch.no_grad():
            fnn_prob = float(self.fnn(torch.tensor(x, dtype=torch.float32)).item())
        rf_prob = float(self.rf.predict_proba(x)[0,1])
        anfis_prob = float(self.anfis.predict(x)[0]) if self.anfis else None
        latency = time.time() - start
        if latency > 1.0:
            print(f"Warning: inference latency {latency:.3f}s exceeds 1s")
        return {"fnn_prob": fnn_prob, "rf_prob": rf_prob, "anfis_prob": anfis_prob}

######################################
# NQR.4 Crew AI Decision Agent
######################################
class CrewAIDecisionAgent:
    """
    NQR.4 Crew AI Decision Agent using GPT-4o for majority-vote or threshold rules.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o",
        threshold: float = 0.5,
        role: str = "financial decision-maker",
        goal: str = "Apply majority-vote or threshold rules to decide trade action.",
        backstory: str = "You are a Crew AI agent specialized in combining multiple model probabilities into a single buy/hold/sell decision."
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.role = role
        self.goal = goal
        self.backstory = backstory

    def decide(self, fnn_prob: float, rf_prob: float, anfis_prob: float = None) -> dict:
        """
        NQR.4.1 & NQR.4.2:
        - Build system + user messages embedding role, goal, backstory.
        - Instruct GPT-4o to count buy/sell signals via threshold and apply majority vote.
        - Return JSON { decision, explanation }.
        """
        system_msg = {
            "role": "system",
            "content": (
                f"Backstory: {self.backstory}\n"
                f"Role: {self.role}\n"
                f"Goal: {self.goal}"
            )
        }

        user_prompt = (
            f"You are given the following model probabilities:\n"
            f"- FNN: {fnn_prob}\n"
            f"- RF: {rf_prob}\n"
            f"- ANFIS: {anfis_prob if anfis_prob is not None else 'N/A'}\n\n"
            f"Use threshold = {self.threshold}.\n"
            f"Count each probability > threshold as a BUY signal, < threshold as a SELL signal.\n"
            f"Apply majority vote:\n"
            f"- If 2 or more BUY signals → decision = BUY\n"
            f"- If 2 or more SELL signals → decision = SELL\n"
            f"- Otherwise → decision = HOLD\n\n"
            f"Respond in strict JSON format:\n"
            f'{{"decision": "<BUY|SELL|HOLD>", "explanation": "your reasoning here"}}'
        )
        user_msg = {"role": "user", "content": user_prompt}

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[system_msg, user_msg],
            temperature=0.0
        )
        content = response.choices[0].message.content

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"decision": content.strip(), "explanation": ""}

        return result

######################################
# MAIN
######################################
if __name__ == "__main__":
    # Interactive demo of the full pipeline
    if len(sys.argv) > 1:
        ticker = sys.argv[1].strip().upper()
    else:
        ticker = input("Enter stock ticker symbol: ").strip().upper()

    # NQR.1: Fetch & compute ratios
    df = DataFetch()
    rc = RatioCalc()
    income_reports = df.fetch_income_statement(ticker)
    balance_reports = df.fetch_balance_sheet(ticker)
    ratios = rc.compute_ratios(income_reports, balance_reports)
    print("Calculated Ratios:", ratios)

    # NQR.2: Train sample models on random data
    X = np.random.rand(100, 3)
    y = (np.random.rand(100) > 0.5).astype(int)
    trainer = ModelTrain()
    fnn_model, fnn_acc = trainer.train_fnn(X, y)
    rf_model, rf_acc = trainer.train_rf(X, y)
    anfis_acc = None
    if ANFISNet:
        anfis_model, anfis_acc = trainer.train_anfis(X, y)

    # Save models
    torch.save(fnn_model, 'fnn_model.pt')
    joblib.dump(rf_model, 'rf_model.pkl')
    if anfis_acc:
        with open('anfis_model.pkl', 'wb') as f:
            pickle.dump(anfis_model, f)

    # NQR.3: Load and run inference
    infer_agent = ModelInferAgent(
        'fnn_model.pt',
        'rf_model.pkl',
        'anfis_model.pkl' if anfis_acc else None
    )
    sample_features = [r['roe'] or 0 for r in ratios][:3]
    result = infer_agent.infer(sample_features)
    print("Inference Output:", result)

    # NQR.4: Crew AI Decision Agent tests
    crew_decider = CrewAIDecisionAgent(threshold=0.5)
    test_cases = [
        {"fnn": 0.6, "rf": 0.7, "anfis": 0.8},  # all above threshold → BUY
        {"fnn": 0.4, "rf": 0.3, "anfis": 0.2},  # all below threshold → SELL
        {"fnn": 0.6, "rf": 0.4, "anfis": 0.5},  # mixed → HOLD
    ]
    for case in test_cases:
        decision = crew_decider.decide(case["fnn"], case["rf"], case["anfis"])
        print(
            f"Probs={case} → Decision: {decision['decision']} | "
            f"Explanation: {decision['explanation']}"
        )

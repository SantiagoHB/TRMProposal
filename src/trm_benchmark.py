import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# 0. Setup Data Directories
RAW_DIR = Path("data/01_raw")
REPORT_DIR = Path("data/08_reporting")
RAW_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load & Preprocess German Credit Dataset (1,000 samples)
print("Loading dataset...")
raw_csv_path = RAW_DIR / "credit_g_raw.csv"
data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
data.frame.to_csv(raw_csv_path, index=False)
print(f"Saved raw dataset to {raw_csv_path}")

X = data.data
y = (data.target == "good").astype(int).values

cat_cols = X.select_dtypes(include=["category", "object"]).columns
num_cols = X.select_dtypes(include=["number"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_proc = preprocessor.fit_transform(X_train).astype(np.float32)
X_test_proc = preprocessor.transform(X_test).astype(np.float32)

input_dim = X_train_proc.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Input features: {input_dim}\n")

results = []

# 2. Baseline 1: XGBoost
t0 = time.time()
xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss")
xgb.fit(X_train_proc, y_train)
xgb_train_time = time.time() - t0

t0 = time.time()
xgb_preds = xgb.predict(X_test_proc)
xgb_inf_ms = ((time.time() - t0) / len(X_test_proc)) * 1000

xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds)
print(f"XGBoost      | Acc: {xgb_acc:.4f} | F1: {xgb_f1:.4f} | Train: {xgb_train_time:.2f}s | Inf: {xgb_inf_ms:.4f} ms/sample")
results.append({
    "Model": "XGBoost",
    "Accuracy": round(xgb_acc, 4),
    "F1_Score": round(xgb_f1, 4),
    "Train_Time_s": round(xgb_train_time, 4),
    "Inference_ms_per_sample": round(xgb_inf_ms, 4),
    "Parameters": None
})

# PyTorch DataLoaders
train_loader = DataLoader(TensorDataset(torch.tensor(X_train_proc), torch.tensor(y_train, dtype=torch.float32)), batch_size=32, shuffle=True)
X_test_t = torch.tensor(X_test_proc).to(device)

# 3. Baseline 2: Standard 2-Layer MLP
class StandardMLP(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# 4. Proposed: Tiny Recursive Model (TRM) for Tabular Data
class TabularTRM(nn.Module):
    def __init__(self, d_in, d_latent=64, steps=4):
        super().__init__()
        self.steps = steps
        self.d_latent = d_latent
        self.input_proj = nn.Linear(d_in, d_latent)
        # Single 2-layer recursive block
        self.recursive_block = nn.Sequential(
            nn.Linear(d_latent * 2 + 1, d_latent),
            nn.LayerNorm(d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
            nn.GELU()
        )
        self.head = nn.Linear(d_latent, 1)

    def forward(self, x):
        b = x.size(0)
        x_emb = self.input_proj(x)
        z = torch.zeros(b, self.d_latent, device=x.device)
        y_hat = torch.zeros(b, 1, device=x.device)
        outputs = []
        for _ in range(self.steps):
            inp = torch.cat([x_emb, z, y_hat], dim=-1)
            z = z + self.recursive_block(inp)  # Residual latent update
            y_hat = self.head(z)
            outputs.append(y_hat.squeeze(-1))
        return outputs  # Deep supervision across all steps

def train_and_eval_torch(model, name, is_trm=False, epochs=40):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            if is_trm:
                outs = model(bx)
                loss = sum(criterion(o, by) for o in outs) / len(outs)
            else:
                loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    train_time = time.time() - t0

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        if is_trm:
            logits = model(X_test_t)[-1]
        else:
            logits = model(X_test_t)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
    inf_ms = ((time.time() - t0) / len(X_test_proc)) * 1000

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"{name:<12} | Acc: {acc:.4f} | F1: {f1:.4f} | Train: {train_time:.2f}s | Inf: {inf_ms:.4f} ms/sample | Params: {params}")
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "F1_Score": round(f1, 4),
        "Train_Time_s": round(train_time, 4),
        "Inference_ms_per_sample": round(inf_ms, 4),
        "Parameters": params
    })

train_and_eval_torch(StandardMLP(input_dim), "Standard MLP", is_trm=False)
train_and_eval_torch(TabularTRM(input_dim, d_latent=64, steps=4), "Tabular TRM", is_trm=True)

# 5. Save Benchmark Report
results_csv_path = REPORT_DIR / "benchmark_results.csv"
pd.DataFrame(results).to_csv(results_csv_path, index=False)
print(f"\nSaved benchmark results to {results_csv_path}")
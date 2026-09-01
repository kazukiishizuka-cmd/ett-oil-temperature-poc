"""深層学習モデル（DLinear / PatchTST）。

どちらも「過去 L ステップの多変量窓」から h ステップ先の油温を予測する。
系列の水準が期間ごとにずれる問題（EDAで見つけた下降トレンド）に対しては、
窓ごとの正規化（RevIN）で対処している。表形式モデル側でΔを予測しているのと同じ発想。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import RANDOM_SEED


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class RevIN(nn.Module):
    """窓ごとの可逆な正規化。

    学習期間と評価期間で系列の水準が違っても、
    窓内の相対的な形だけを見て予測できるようにする。
    """

    def __init__(self, n_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self.mean_ = x.mean(dim=1, keepdim=True).detach()
            self.std_ = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean_) / self.std_
            return x * self.weight + self.bias
        if mode == "denorm":
            x = (x - self.bias) / (self.weight + self.eps)
            return x * self.std_ + self.mean_
        raise ValueError(mode)


class MovingAvg(nn.Module):
    """系列をトレンド成分に均す移動平均。DLinearの分解に使う。"""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):  # x: (B, L, C)
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class DLinear(nn.Module):
    """系列をトレンドと残差に分解し、それぞれを線形写像するモデル。

    Transformer系より単純だが、長期予測ベンチマークで強い結果を出すことが
    報告されている（Zeng et al., 2023）。本PoCでは「複雑さが要るのか」の対照として置く。
    """

    def __init__(self, seq_len: int, n_features: int, kernel_size: int = 25, target_idx: int = -1):
        super().__init__()
        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(seq_len, 1)
        self.linear_season = nn.Linear(seq_len, 1)
        self.revin = RevIN(n_features)
        self.target_idx = target_idx

    def forward(self, x):  # (B, L, C) -> (B, 1)
        x = self.revin(x, "norm")
        trend = self.decomp(x)
        season = x - trend
        t = self.linear_trend(trend.permute(0, 2, 1))      # (B, C, 1)
        s = self.linear_season(season.permute(0, 2, 1))
        out = (t + s).permute(0, 2, 1)                      # (B, 1, C)
        out = self.revin(out, "denorm")
        return out[:, 0, self.target_idx]


class PatchTST(nn.Module):
    """系列を固定長パッチに切ってTransformerに通すモデル（Nie et al., 2023）。

    チャネル独立（各変数を同じエンコーダで別々に処理）を採用している。
    """

    def __init__(self, seq_len: int, n_features: int, patch_len: int = 16, stride: int = 8,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.15,
                 target_idx: int = -1):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.target_idx = target_idx
        self.revin = RevIN(n_features)
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model * self.n_patches, 1)

    def forward(self, x):  # (B, L, C) -> (B,)
        x = self.revin(x, "norm")
        b, l, c = x.shape
        z = x.permute(0, 2, 1)                                  # (B, C, L)
        # unfold はビューを返す。MPS ではビューのまま reshape すると
        # バッファ確保に失敗することがあるので、明示的に連続メモリに落とす。
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride).contiguous()
        z = z.reshape(b * c, self.n_patches, self.patch_len)
        z = self.embed(z) + self.pos
        z = self.encoder(self.dropout(z))
        z = z.reshape(b * c, -1)
        out = self.head(z).reshape(b, c, 1).permute(0, 2, 1)    # (B, 1, C)
        out = self.revin(out, "denorm")
        return out[:, 0, self.target_idx]


# --- 学習ループ -------------------------------------------------------------

def make_windows(values: np.ndarray, target: np.ndarray, seq_len: int, horizon: int,
                 valid_flags: np.ndarray):
    """(B, L, C) の入力窓と、h ステップ先の目的値を作る。

    窓や予測先に欠測区間が混ざるサンプルは valid_flags で落とす。
    """
    n = len(values)
    idx = []
    for t in range(seq_len - 1, n - horizon):
        if valid_flags[t - seq_len + 1: t + 1].all() and valid_flags[t + horizon]:
            idx.append(t)
    idx = np.asarray(idx)
    X = np.stack([values[i - seq_len + 1: i + 1] for i in idx]).astype(np.float32)
    y = target[idx + horizon].astype(np.float32)
    return X, y, idx


class DeepForecaster:
    """DLinear / PatchTST を共通の手順で学習・予測するラッパー。"""

    def __init__(self, kind: str, seq_len: int = 336, horizon: int = 1,
                 max_epochs: int = 40, patience: int = 6, batch_size: int = 256,
                 lr: float = 1e-3, verbose: bool = False, device_override=None):
        self.kind = kind
        self.seq_len = seq_len
        self.horizon = horizon
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.device_override = device_override
        self.name = kind

    def _build(self, n_features: int) -> nn.Module:
        torch.manual_seed(RANDOM_SEED)
        if self.kind == "DLinear":
            return DLinear(self.seq_len, n_features)
        if self.kind == "PatchTST":
            return PatchTST(self.seq_len, n_features)
        raise ValueError(self.kind)

    def _resolve_device(self) -> torch.device:
        """device の決定。強制指定があればそれに従う。"""
        if self.device_override is not None:
            return torch.device(self.device_override)
        return get_device()

    def fit(self, X_tr, y_tr, X_val=None, y_val=None):
        device = self._resolve_device()
        self.device = device
        self.model = self._build(X_tr.shape[2]).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.L1Loss()  # 表形式モデルのobjective=l1と揃える
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=self.max_epochs * max(1, len(X_tr) // self.batch_size + 1))

        dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=self.batch_size, shuffle=True, drop_last=False)
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            xv = torch.from_numpy(X_val).to(device)
            yv = torch.from_numpy(y_val).to(device)

        best, best_state, bad = float("inf"), None, 0
        for ep in range(self.max_epochs):
            self.model.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                try:
                    sched.step()
                except ValueError:
                    pass
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    vl = float(nn.functional.l1_loss(self._predict_batched(xv), yv))
                if vl < best - 1e-4:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
                if self.verbose:
                    print(f"    ep{ep:02d} val_MAE={vl:.4f}")
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_ = best
        return self

    #: 推論時のバッチ。PatchTSTはチャネル独立なので実効バッチが変数の数だけ膨らむ。
    #: MPSは大きすぎる中間テンソルでバッファ確保に失敗するため、控えめに固定する。
    predict_batch_size = 256

    def _predict_batched(self, x: torch.Tensor) -> torch.Tensor:
        bs = self.predict_batch_size
        outs = []
        for i in range(0, len(x), bs):
            outs.append(self.model(x[i:i + bs]))
        return torch.cat(outs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X).to(self.device)
            return self._predict_batched(x).cpu().numpy()

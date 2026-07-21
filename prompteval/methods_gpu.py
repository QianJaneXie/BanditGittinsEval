"""
PyTorch (GPU-capable) counterparts to the sklearn models used in BAI:
LogReg, ExtendedRaschModel (PromptEval / "pe"), and MLP + CV.

Numerical results will not match sklearn exactly; optimization differs.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from .methods import GenXY, check_multicolinearity, sigmoid
except ImportError:
    from methods import GenXY, check_multicolinearity, sigmoid


def resolve_torch_device(device: str) -> torch.device:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device: {device!r} (use auto, cpu, or cuda)")


class TorchLogisticRegression:
    """
    Same role as methods.LogisticRegression: fit (X, y) -> attribute mu (coef vector).
    """

    def __init__(
        self,
        reg: float = 1e2,
        device: str = "auto",
        epochs: int = 800,
        lr: float = 0.5,
        fit_log_interval: int = 0,
    ):
        self.reg = float(reg)
        self.device = device
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.fit_log_interval = int(fit_log_interval)
        self.mu: Optional[np.ndarray] = None

    def fit(self, X_np: np.ndarray, y_np: np.ndarray) -> "TorchLogisticRegression":
        dev = resolve_torch_device(self.device)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        X_np = np.asarray(X_np, dtype=np.float64)
        y_np = np.asarray(y_np, dtype=np.float64).ravel()
        if np.var(y_np) == 0:
            y_copy = y_np.copy()
            local_state = np.random.RandomState(0)
            ind = local_state.choice(len(y_copy))
            y_copy[ind] = 1 - np.median(y_copy)
            y_np = y_copy

        X = torch.as_tensor(X_np, dtype=torch.float32, device=dev)
        y = torch.as_tensor(y_np, dtype=torch.float32, device=dev).unsqueeze(1)
        n_in = X.shape[1]
        lin = nn.Linear(n_in, 1, bias=False).to(dev)
        # sklearn L2 with C=reg: penalty (1/(2C))||w||^2
        opt = torch.optim.AdamW(lin.parameters(), lr=self.lr, weight_decay=1.0 / (2.0 * self.reg))
        loss_fn = nn.BCEWithLogitsLoss()
        lin.train()
        log_every = self.fit_log_interval
        for ep in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(lin(X), y)
            loss.backward()
            opt.step()
            if log_every > 0 and (ep % log_every == 0 or ep == self.epochs - 1):
                with torch.no_grad():
                    loss_v = float(loss_fn(lin(X), y).item())
                print(
                    f"TorchLogisticRegression epoch {ep + 1}/{self.epochs} loss={loss_v:.6f}",
                    flush=True,
                )
        self.mu = lin.weight.detach().squeeze(0).cpu().numpy()
        return self


class _SmallMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 30):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    sig = np.where(sig < 1e-8, 1.0, sig)
    return (X - mu) / sig, mu, sig


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (X - mu) / sig


def _mlp_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    lr: float,
    dev: torch.device,
    cv: int,
    max_iter: int,
    hidden: int,
    seed: int,
) -> float:
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, cv)
    scores: list[float] = []
    for fi in range(cv):
        val_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(cv) if j != fi])
        if train_idx.size < 2 or val_idx.size < 1:
            continue
        X_tr, mu, sig = _standardize_fit(X[train_idx])
        X_va = _standardize_apply(X[val_idx], mu, sig)
        y_tr = y[train_idx]
        y_va = y[val_idx]

        Xt = torch.as_tensor(X_tr, dtype=torch.float32, device=dev)
        yt = torch.as_tensor(y_tr, dtype=torch.float32, device=dev).unsqueeze(1)
        Xv = torch.as_tensor(X_va, dtype=torch.float32, device=dev)
        yv = torch.as_tensor(y_va, dtype=torch.float32, device=dev).unsqueeze(1)

        model = _SmallMLP(X_tr.shape[1], hidden=hidden).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        def l2_term() -> torch.Tensor:
            s = torch.zeros((), device=dev)
            for p in model.parameters():
                s = s + (p**2).sum()
            return alpha * s

        loss_fn = nn.BCEWithLogitsLoss()
        torch.manual_seed(seed + fi)
        model.train()
        for _ in range(max_iter):
            opt.zero_grad(set_to_none=True)
            logits = model(Xt)
            loss = loss_fn(logits, yt) + l2_term()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(Xv)) >= 0.5).float()
            acc = (pred == yv).float().mean().item()
        scores.append(acc)
    return float(np.mean(scores)) if scores else 0.0


def _mlp_fit_full(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    lr: float,
    dev: torch.device,
    max_iter: int,
    hidden: int,
    seed: int,
) -> _SmallMLP:
    Xs, mu, sig = _standardize_fit(X)
    Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)
    model = _SmallMLP(X.shape[1], hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    def l2_term() -> torch.Tensor:
        s = torch.zeros((), device=dev)
        for p in model.parameters():
            s = s + (p**2).sum()
        return alpha * s

    torch.manual_seed(seed)
    model.train()
    for _ in range(max_iter):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(Xt), yt) + l2_term()
        loss.backward()
        opt.step()
    model.eval()
    model._mu_np = mu  # type: ignore[attr-defined]
    model._sig_np = sig  # type: ignore[attr-defined]
    return model


class MLPClassifierWithCVTorch:
    """
    PyTorch MLP + grid search (lighter than sklearn's full 5x4 grid × 200-iter CV for speed).
    Pass explicit alphas / learning_rate_inits to match sklearn more closely (slower).
    """

    def __init__(
        self,
        device: str = "auto",
        alphas: Optional[np.ndarray] = None,
        learning_rate_inits: Optional[np.ndarray] = None,
        hidden_layer_sizes: tuple[int, ...] = (30,),
        max_iter: int = 120,
        cv: int = 3,
        random_state: int = 42,
        cv_max_iter: int = 40,
    ):
        self.device = device
        # Default: coarser grid + fewer CV epochs than sklearn GridSearchCV (much faster across BAI phases).
        self.alphas = np.logspace(-2, 0, 3) if alphas is None else np.asarray(alphas)
        self.learning_rate_inits = (
            np.logspace(-2.5, -1.5, 3) if learning_rate_inits is None else np.asarray(learning_rate_inits)
        )
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.cv_max_iter = cv_max_iter
        self.cv = cv
        self.random_state = random_state
        self.best_model: Optional[_SmallMLP] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifierWithCVTorch":
        dev = resolve_torch_device(self.device)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        cut = self.cv
        if min((y == 0).sum(), (y == 1).sum()) < cut:
            y_copy = copy.deepcopy(y)
            local_state = np.random.RandomState(0)
            ind = local_state.choice(len(y_copy), cut)
            y_copy[ind] = 1 - np.median(y_copy)
            y_fit = y_copy
        else:
            y_fit = y

        hidden = int(self.hidden_layer_sizes[0]) if self.hidden_layer_sizes else 30
        best_score = -1.0
        best_a, best_lr = float(self.alphas[0]), float(self.learning_rate_inits[0])
        for a in self.alphas:
            for lr in self.learning_rate_inits:
                sc = _mlp_cv_score(
                    X,
                    y_fit,
                    float(a),
                    float(lr),
                    dev,
                    self.cv,
                    self.cv_max_iter,
                    hidden,
                    self.random_state,
                )
                if sc > best_score:
                    best_score = sc
                    best_a, best_lr = float(a), float(lr)

        self.best_model = _mlp_fit_full(X, y_fit, best_a, best_lr, dev, self.max_iter, hidden, self.random_state + 999)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Call fit first")
        dev = next(self.best_model.parameters()).device
        X = np.asarray(X, dtype=np.float64)
        mu = getattr(self.best_model, "_mu_np")
        sig = getattr(self.best_model, "_sig_np")
        Xs = _standardize_apply(X, mu, sig)
        Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
        with torch.no_grad():
            p1 = torch.sigmoid(self.best_model(Xt)).squeeze(-1).cpu().numpy()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class LogRegTorch:
    """GPU/torch version of methods.LogReg."""

    def __init__(self, device: str = "auto", fit_log_interval: int = 0):
        self.device = device
        self.fit_log_interval = int(fit_log_interval)

    def fit(self, seen_items, Y, X=None):
        if type(X) != np.ndarray:
            self.X = np.eye(Y.shape[0])
        else:
            self.X = X
        self.x_dim = self.X.shape[1]
        self.Z = np.eye(Y.shape[1])
        self.z_dim = self.Z.shape[1]
        self.n_formats, self.n_examples = seen_items.shape
        features, labels = GenXY(seen_items, Y, self.X, self.Z)
        features = features[:, : self.x_dim]
        if type(X) == np.ndarray:
            features = np.hstack((features, np.ones((features.shape[0], 1))))

        self.rasch_model = TorchLogisticRegression(
            reg=1e2, device=self.device, fit_log_interval=self.fit_log_interval
        )
        self.rasch_model.fit(features, labels)

        self.gammas = self.rasch_model.mu[: self.x_dim]
        self.thetas = self.X @ self.gammas
        self.logits = self.thetas[:, None] + self.rasch_model.mu[-1] + np.zeros(Y.shape)
        self.seen_examples = seen_items
        self.Y = Y

    def get_Y_hat(self):
        P_hat = sigmoid(self.logits)
        Y_hat = np.zeros(self.seen_examples.shape)
        Y_hat[self.seen_examples] = self.Y[self.seen_examples]
        Y_hat[~self.seen_examples] = P_hat[~self.seen_examples]
        return Y_hat


class ExtendedRaschModelTorch:
    """GPU/torch version of methods.ExtendedRaschModel."""

    def __init__(self, device: str = "auto"):
        self.device = device

    def fit(self, seen_examples, Y, X=None, Z=None):
        self.seen_examples = seen_examples
        self.Y = Y
        if type(X) != np.ndarray:
            self.X = np.eye(Y.shape[0])
        else:
            self.X = X
            check_multicolinearity(X)
        self.x_dim = self.X.shape[1]
        if type(Z) != np.ndarray:
            self.Z = np.eye(Y.shape[1])
        else:
            self.Z = Z
            check_multicolinearity(Z)
        self.z_dim = self.Z.shape[1]
        self.n_formats, self.n_examples = seen_examples.shape
        features, labels = GenXY(seen_examples, Y, self.X, self.Z)

        if type(X) != np.ndarray and type(Z) != np.ndarray:
            features = features[:, :-1]
        elif type(X) != np.ndarray or type(Z) != np.ndarray:
            pass
        else:
            features = np.hstack((features, np.ones((features.shape[0], 1))))

        self.rasch_model = TorchLogisticRegression(reg=1e2, device=self.device)
        self.rasch_model.fit(features, labels)

        self.gammas = self.rasch_model.mu[: self.x_dim]
        self.thetas = self.X @ self.gammas
        self.psi = self.rasch_model.mu[self.x_dim :]

        if type(X) != np.ndarray and type(Z) != np.ndarray:
            self.betas = np.hstack((self.psi, np.array([0])))
            self.logits = self.thetas[:, None] + self.betas[None, :]
        elif type(X) != np.ndarray or type(Z) != np.ndarray:
            self.betas = self.Z @ self.psi
            self.logits = self.thetas[:, None] + self.betas[None, :]
        else:
            self.betas = self.Z @ self.psi[:-1]
            self.logits = self.thetas[:, None] + self.betas[None, :] + self.psi[-1]

    def get_Y_hat(self):
        P_hat = sigmoid(self.logits)
        Y_hat = np.zeros(self.seen_examples.shape)
        Y_hat[self.seen_examples] = self.Y[self.seen_examples]
        Y_hat[~self.seen_examples] = P_hat[~self.seen_examples]
        return Y_hat


class MLPTorch:
    """GPU/torch version of bai_evaluation.MLP."""

    def __init__(self, device: str = "auto"):
        self.device = device

    def fit(self, seen_items, Y, X=None):
        if type(X) != np.ndarray:
            self.X = np.eye(Y.shape[0])
        else:
            self.X = X
        self.x_dim = self.X.shape[1]
        self.Z = np.eye(Y.shape[1])
        self.z_dim = self.Z.shape[1]
        self.n_formats, self.n_examples = seen_items.shape
        features, labels = GenXY(seen_items, Y, self.X, self.Z)
        features = features[:, : self.x_dim]
        self.model = MLPClassifierWithCVTorch(device=self.device)
        self.model.fit(features, labels)
        self.thetas = self.model.predict_proba(self.X)[:, 1]

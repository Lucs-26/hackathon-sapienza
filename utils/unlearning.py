import copy
import numpy as np
import torch
import torch.nn.functional as F


def _fisher_diagonal(model, X, y, batch_size=256, max_batches=None, device='cpu'):
    """Diagonal Fisher information: mean squared gradient per parameter."""
    model.eval()  # keep BatchNorm statistics frozen
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    n_batches = 0

    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        yb = torch.tensor(y[i:i + batch_size], dtype=torch.float32, device=device)
        if len(xb) < 2:
            continue

        model.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        n_batches += 1

        if max_batches and n_batches >= max_batches:
            break

    for n in fisher:
        fisher[n] /= max(n_batches, 1)
    model.zero_grad()
    return fisher


def ssd(model, X_forget, y_forget, X_retain, y_retain,
        alpha=10.0, lambda_damp=1.0, retain_batches=40, device='cpu'):
    """Selective Synaptic Dampening (Foster et al., 2023).

    Dampens weights that matter much more for Df than for Dr.
    No retraining, no noise, no gradient descent.
    """
    m = copy.deepcopy(model).to(device)

    f_forget = _fisher_diagonal(m, X_forget, y_forget, device=device)
    f_retain = _fisher_diagonal(m, X_retain, y_retain,
                                max_batches=retain_batches, device=device)

    eps = 1e-12
    with torch.no_grad():
        for n, p in m.named_parameters():
            ratio = f_forget[n] / (f_retain[n] + eps)
            selected = ratio > alpha
            if selected.any():
                # dampening factor, capped at 1 so weights are never amplified
                beta = torch.clamp(lambda_damp * f_retain[n] / (f_forget[n] + eps), max=1.0)
                p[selected] = p[selected] * beta[selected]
    return m


def fisher_forgetting(model, X_retain, y_retain, sigma=1e-3,
                      retain_batches=40, device='cpu'):
    """Gaussian noise scaled by inverse Fisher computed on Dr only."""
    m = copy.deepcopy(model).to(device)
    f_retain = _fisher_diagonal(m, X_retain, y_retain,
                                max_batches=retain_batches, device=device)
    eps = 1e-8
    with torch.no_grad():
        for n, p in m.named_parameters():
            noise = torch.randn_like(p) * sigma / torch.sqrt(f_retain[n] + eps)
            p.add_(noise)
    return m


def gradient_ascent(model, X_forget, y_forget, lr=1e-4, steps=5,
                    batch_size=256, device='cpu'):
    """Maximize loss on Df. High risk of catastrophic unlearning."""
    m = copy.deepcopy(model).to(device)
    m.eval()
    opt = torch.optim.SGD(m.parameters(), lr=lr)

    for _ in range(steps):
        idx = np.random.choice(len(X_forget), min(batch_size, len(X_forget)), replace=False)
        xb = torch.tensor(X_forget[idx], dtype=torch.float32, device=device)
        yb = torch.tensor(y_forget[idx], dtype=torch.float32, device=device)
        opt.zero_grad()
        loss = -F.binary_cross_entropy_with_logits(m(xb), yb)
        loss.backward()
        opt.step()
    return m


def finetune_retain(model, X_retain, y_retain, lr=1e-4, steps=50,
                    batch_size=256, device='cpu'):
    """Weak baseline: keep training on Dr only, hope Df fades."""
    m = copy.deepcopy(model).to(device)
    m.eval()
    opt = torch.optim.Adam(m.parameters(), lr=lr)

    for _ in range(steps):
        idx = np.random.choice(len(X_retain), min(batch_size, len(X_retain)), replace=False)
        xb = torch.tensor(X_retain[idx], dtype=torch.float32, device=device)
        yb = torch.tensor(y_retain[idx], dtype=torch.float32, device=device)
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(m(xb), yb)
        loss.backward()
        opt.step()
    return m
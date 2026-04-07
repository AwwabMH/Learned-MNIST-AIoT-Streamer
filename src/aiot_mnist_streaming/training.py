from __future__ import annotations

import copy
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass(slots=True)
class TrainingResult:
    history: pd.DataFrame
    best_state_dict: dict


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def fit_model(model, train_loader, test_loader, epochs, name, training_config, device):
    optimizer = optim.Adam(model.parameters(), lr=training_config.lr, weight_decay=training_config.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=training_config.label_smoothing)
    scaler = torch.cuda.amp.GradScaler() if (training_config.use_amp and device.type == "cuda") else None
    history_rows = []
    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        te_loss, te_acc = evaluate(model, test_loader, device)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "test_loss": te_loss,
                "test_acc": te_acc,
            }
        )
        print(f"[{name}] epoch {epoch:02d} | train_acc={tr_acc:.4f} | test_acc={te_acc:.4f}")
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return TrainingResult(history=pd.DataFrame(history_rows), best_state_dict=best_state)

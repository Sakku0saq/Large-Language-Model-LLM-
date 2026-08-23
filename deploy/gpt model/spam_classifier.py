"""
Fine-tunes a classification head on top of pretrained GPT-2 to detect spam,
following the same recipe as the notebook: freeze the base model, unfreeze
only the last transformer block + final norm, add a 2-class linear head,
and train on the SMS Spam Collection dataset.
"""
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, pad_token_id=50256):
        self.encoded_texts = [tokenizer.encode(t) for t in texts]
        self.max_length = max_length
        self.encoded_texts = [
            et[:max_length] for et in self.encoded_texts
        ]
        self.encoded_texts = [
            et + [pad_token_id] * (max_length - len(et)) for et in self.encoded_texts
        ]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def load_sms_spam_data(seed=123):
    """Loads and balances the SMS Spam Collection dataset from the HF Hub."""
    from datasets import load_dataset

    ds = load_dataset("sms_spam", trust_remote_code=True)["train"]
    texts = [ex["sms"].strip() for ex in ds]
    labels = [ex["label"] for ex in ds]  # 0 = ham, 1 = spam

    random.seed(seed)
    spam_idx = [i for i, l in enumerate(labels) if l == 1]
    ham_idx = [i for i, l in enumerate(labels) if l == 0]
    ham_idx = random.sample(ham_idx, len(spam_idx))
    keep = spam_idx + ham_idx
    random.shuffle(keep)

    texts = [texts[i] for i in keep]
    labels = [labels[i] for i in keep]

    n = len(texts)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    return (
        (texts[:train_end], labels[:train_end]),
        (texts[train_end:val_end], labels[train_end:val_end]),
        (texts[val_end:], labels[val_end:]),
    )


def calc_accuracy(loader, model, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_batch, target_batch in loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)[:, -1, :]
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == target_batch).sum().item()
            total += target_batch.size(0)
    return correct / total if total else 0.0


def build_and_train_classifier(base_model, cfg, tokenizer, device,
                                num_epochs=3, batch_size=8, max_length=64,
                                progress_callback=None):
    """
    base_model: a GPTModel already loaded with pretrained GPT-2 weights.
    Returns: (model, tokenizer_max_length, test_accuracy)
    """
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = \
        load_sms_spam_data()

    train_ds = SpamDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = SpamDataset(val_texts, val_labels, tokenizer, max_length)
    test_ds = SpamDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = base_model
    for p in model.parameters():
        p.requires_grad = False

    model.out_head = nn.Linear(cfg["emb_dim"], 2)
    for p in model.trf_blocks[-1].parameters():
        p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True

    model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=5e-5, weight_decay=0.1
    )

    total_steps = num_epochs * len(train_loader)
    step = 0
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()
            logits = model(input_batch)[:, -1, :]
            loss = torch.nn.functional.cross_entropy(logits, target_batch)
            loss.backward()
            optimizer.step()
            step += 1
            if progress_callback:
                progress_callback(step, total_steps, loss.item())

    test_accuracy = calc_accuracy(test_loader, model, device)
    return model, max_length, test_accuracy


def classify_text(text, model, tokenizer, device, max_length, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)[:max_length]
    input_ids = input_ids + [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    label = torch.argmax(probs).item()
    return ("spam" if label == 1 else "not spam"), probs[1].item()

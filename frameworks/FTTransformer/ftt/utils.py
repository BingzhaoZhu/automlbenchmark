import torch
from tqdm.auto import tqdm
import enum
from typing import List, Optional
from torch import Tensor, nn
import math


def pretrain_epoch(model, criterion, train_loader, optimizer, device, epoch):
    model.train()
    epoch_loss = 0.0
    batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for anchor_cat, anchor_con, _ in batch:
        anchor_cat, anchor_con = anchor_cat.to(device), anchor_con.to(device)

        # reset gradients
        optimizer.zero_grad()

        # get embeddings
        emb_anchor, emb_positive = model(anchor_cat, anchor_con)

        # compute loss
        loss = criterion(emb_anchor, emb_positive)
        loss.backward()

        # update Models weights
        optimizer.step()

        # log progress
        epoch_loss += anchor_cat.size(0) * loss.item()
        batch.set_postfix({"loss": loss.item()})

    return epoch_loss / len(train_loader.dataset)


def dataset_embeddings(model, loader, device):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for anchor_cat, anchor_con, _ in tqdm(loader):
            anchor_cat = anchor_cat.to(device)
            anchor_con = anchor_con.to(device)
            embeddings.append(model.get_embeddings(anchor_cat, anchor_con))

    embeddings = torch.cat(embeddings).to('cpu').numpy()

    return embeddings


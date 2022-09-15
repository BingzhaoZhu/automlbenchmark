from .ft_transformer import FTTransformer_
from tqdm.auto import tqdm
import torch
from torch.optim import Adam
import torch.nn as nn


class FTTransformer():
    def __init__(self, cat_dims=None,
                 n_con=None,
                 num_classes=None,
                 is_classification=None,
                 device="cuda"):
        self.device = device
        self.model = FTTransformer_(
            prefix="FTTransformer",
            num_categories=cat_dims,
            in_features=n_con,
            d_token=32,
            n_blocks=8,
            ffn_d_hidden=128,
            train_status="finetune",
            num_classes=num_classes,
        ).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    def fit(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0.0
        batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for anchor_cat, anchor_con, y in batch:
            anchor_cat, anchor_con, y = anchor_cat.to(self.device), anchor_con.to(self.device), y.to(self.device)

            # reset gradients
            self.optimizer.zero_grad()

            # get embeddings
            emb_anchor = self.model(anchor_cat, anchor_con)

            # compute loss
            loss = self.criterion(emb_anchor, y)
            loss.backward()

            # update Models weights
            self.optimizer.step()

            # log progress
            epoch_loss += anchor_cat.size(0) * loss.item()
            batch.set_postfix({"loss": loss.item()})

        return epoch_loss / len(train_loader.dataset)


    def validate(self, valid_loader):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for anchor_cat, anchor_con, y in tqdm(valid_loader):
                anchor_cat, anchor_con, y = anchor_cat.to(self.device), anchor_con.to(self.device), y.to(self.device)
                yhat = self.model(anchor_cat, anchor_con)
                loss = self.criterion(yhat, y)
                epoch_loss += anchor_cat.size(0) * loss.item()
        return epoch_loss / len(valid_loader.dataset)


    def predict(self, test_loader):
        self.model.eval()
        yhat_all, y_all = [], []
        with torch.no_grad():
            for anchor_cat, anchor_con, y in tqdm(test_loader):
                anchor_cat, anchor_con, y = anchor_cat.to(self.device), anchor_con.to(self.device), y.to(self.device)
                yhat = self.model(anchor_cat, anchor_con)
                yhat_all.append(yhat)
                y_all.append(y)
        yhat_all = torch.cat(yhat_all).to('cpu').numpy()
        y_all = torch.cat(y_all).to('cpu').numpy()
        return yhat_all, y_all

from .ft_transformer import FTTransformer_
from tqdm.auto import tqdm
import torch
from torch.optim import Adam
import torch.nn as nn
from .loss import NTXent

class FTTransformer():
    def __init__(self, cat_dims=None,
                 n_con=None,
                 num_classes=None,
                 is_classification=None,
                 d_token=128,
                 n_blocks=8,
                 ffn_d_hidden=128,
                 device="cuda"):
        self.device = device
        self.model = FTTransformer_(
            prefix="FTTransformer",
            num_categories=cat_dims,
            in_features=n_con,
            d_token=d_token,
            n_blocks=n_blocks,
            ffn_d_hidden=ffn_d_hidden,
            train_status="finetune",
            num_classes=num_classes,
        ).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
        self.pretrain_criterion = NTXent(device=self.device)
        self.random_samples_con, self.random_samples_cat = None, None

    def pretrain(self, pretrain_loader, epoch):
        self.model.train()
        self.model.set_train_status('pretrain')
        epoch_loss = 0.0
        batch = tqdm(pretrain_loader, desc=f"Pretrain Epoch {epoch}", leave=False)

        for ori_cat, ori_con, _ in batch:
            cor_cat, cor_con = self.corruption(ori_cat, 'cat').type(torch.int64), self.corruption(ori_con, 'con')
            ori_cat, ori_con = ori_cat.to(self.device), ori_con.to(self.device)
            cor_cat, cor_con = cor_cat.to(self.device), cor_con.to(self.device)

            # reset gradients
            self.optimizer.zero_grad()

            # get embeddings
            ori_emb = self.model(ori_cat, ori_con)
            cor_emb = self.model(cor_cat, cor_con)

            # compute loss
            loss = self.pretrain_criterion(ori_emb, cor_emb)
            loss.backward()

            # update Models weights
            self.optimizer.step()

            # log progress
            epoch_loss += ori_cat.size(0) * loss.item()
            batch.set_postfix({"loss": loss.item()})

        return epoch_loss / len(pretrain_loader.dataset)

    def get_pretrain_embedding(self, data_loader):
        self.model.eval()
        self.model.set_train_status('pretrain')
        yhat_all = []
        with torch.no_grad():
            for anchor_cat, anchor_con, _ in tqdm(data_loader):
                anchor_cat, anchor_con = anchor_cat.to(self.device), anchor_con.to(self.device)
                yhat = self.model.get_embeddings(anchor_cat, anchor_con)
                yhat_all.append(yhat)
        yhat_all = torch.cat(yhat_all).to('cpu').numpy()
        return yhat_all


    def fit(self, train_loader, epoch):
        self.model.train()
        self.model.set_train_status('finetune')
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
        self.model.set_train_status('finetune')
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
        self.model.set_train_status('finetune')
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


    def corruption(self, anchor, type, corruption_rate=0.6):
        batch_size, m = anchor.size()

        last_batch = self.random_samples_con if type == 'con' else self.random_samples_cat
        if last_batch and last_batch.size(1) == batch_size:
            random_sample = last_batch
        else:
            random_sample = anchor
        indices = torch.argsort(torch.rand(*random_sample.shape), dim=0)
        random_sample = random_sample[indices, torch.arange(m).unsqueeze(0)]

        # 1: create a mask of size (batch size, m) where for each sample we set the
        # jth column to True at random, such that corruption_len / m = corruption_rate
        # 3: replace x_1_ij by x_2_ij where mask_ij is true to build x_corrupted

        corruption_mask = torch.zeros_like(anchor, dtype=torch.bool)
        corruption_len = int(corruption_rate * m)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[:corruption_len]
            corruption_mask[i, corruption_idx] = True

        positive = torch.where(corruption_mask, random_sample, anchor)

        if type == 'con':
            self.random_samples_con = anchor
        else:
            self.random_samples_cat = anchor

        return positive

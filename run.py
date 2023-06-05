import torch
import torch.nn as nn
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertConfig
from transformers import BertTokenizerFast

import json
from tqdm import tqdm
import os
import sys
import random

from lr_scheduler import ReduceLROnPlateau, Lookahead, RAdam
from model import *
from utils import *
from config import *
from loader import *


def train():

    train_data = load_lcqmc(cfg.train_path)
    random.seed(cfg.random_seed)
    random.shuffle(train_data)
    dev_data = load_lcqmc(cfg.dev_path)

    collator = Collator(cfg, tokenizer)
    data_loader = DataLoader(train_data, collate_fn=collator, batch_size=cfg.batch_size, num_workers=0)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    t_total = len(data_loader) * cfg.epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr, eps=cfg.min_num)
    optimizer = Lookahead(optimizer, 5, 1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, t_total // cfg.epochs * 1, 1, eta_min=cfg.min_lr, last_epoch=-1)

    ema = EMA(model.parameters(), decay=0.999)

    best_v = - 1
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=data_loader.__len__(), desc="train", ncols=80) as t:
            for i, batch in enumerate(data_loader):
                batch = [d.to(cfg.device) for d in batch]
                batch_token, batch_mask,batch_label= batch
                logits,neg_logits = model(batch_token, batch_mask)
                loss = cosent_loss(logits, batch_label)
                cl_loss = sscl_loss(logits, neg_logits)
                logits2,_ = model(batch_token, batch_mask)
                rdrop_loss = compute_kl_loss(logits, logits2)
                loss = loss + cl_loss*0.2 + rdrop_loss*0.1
                loss.backward()
                pgd = PGD(model)
                pgd.backup_grad()
                for k in range(3):
                    pgd.attack(is_first_attack=(k==0))
                    if k!=2:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    adv_logits,_= model(batch_token, batch_mask)
                    adv_loss = cosent_loss(adv_logits, batch_label)
                    adv_loss.backward()
                pgd.restore()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                ema.update(model.parameters())
                model.zero_grad()
                t.set_postfix(loss="%.4lf" % (loss.cpu().item()))
                t.update(1)
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
        print("")
        corrcoef = evaluate(model, cfg, dev_data, tokenizer)
        scheduler.step()
        if corrcoef > best_v:
            best_v = corrcoef
            if isinstance(model, nn.DataParallel):
                model_stat_dict = model.module.state_dict()
            else:
                model_stat_dict = model.state_dict()
            state = {'epoch': epoch, 'state_dict': model_stat_dict}
            torch.save(state, cfg.save_path)
        ema.restore(model.parameters())


def test():
    states = torch.load(cfg.save_path)
    state = states['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    test_data = load_lcqmc(cfg.test_path)
    corrcoef = evaluate(model, cfg, test_data, tokenizer)




if __name__ == '__main__':

    cfg = Config()
    tokenizer = BertTokenizerFast.from_pretrained(cfg.bert_path, do_lower_case=True, add_special_tokens=True)
    config = BertConfig.from_pretrained(cfg.config_path)
    config.bert_path = cfg.bert_path
    model = SentenceRepresenation.from_pretrained(pretrained_model_name_or_path=cfg.checkpoint_path, config=config)
    model.to(cfg.device)

    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
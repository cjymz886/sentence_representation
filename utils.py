from tqdm import tqdm
import json
import torch
import numpy as np
import re
import scipy.stats
import torch.nn.functional as F
from scipy.optimize import minimize
from loader import re_match, sequence_padding



def compute_kl_loss(p,q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p,dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q,dim=-1), F.softmax(p,dim=-1), reduction='none')
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss+q_loss) / 2
    return loss


def cosent_loss(y_pred,y_true):
    y_true = y_true[::2]
    norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
    y_pred = y_pred / norms
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_true = y_true[:, None] < y_true[None, :]
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0) 
    return torch.logsumexp(y_pred, dim=0)



def sscl_loss(y_pred, y_neg):
    tmp = 0.07
    fcos = torch.nn.CosineSimilarity(dim=-1)
    idxs = torch.arange(0, y_pred.size()[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = idxs_1==idxs_2
    y_true = y_true.float().cuda()
    p_sim = fcos(y_pred.unsqueeze(1), y_pred.unsqueeze(0))  / tmp
    n_sim = fcos(y_pred.unsqueeze(1), y_neg.unsqueeze(0))  / tmp
    pp_sim = p_sim - (torch.eye(y_pred.size()[0]) * 1e12).to('cuda')
    fm = torch.logsumexp(n_sim, dim=0).unsqueeze(-1)
    pm = torch.logsumexp(pp_sim,dim=0).unsqueeze(-1)
    loss =(-(p_sim-(fm+pm))*y_true).mean()
    return loss




def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation




def evaluate(model, cfg, data, tokenizer):
    def convert_to_ids(e_list):
        input_ids, input_mask = [], []
        for e in e_list:
            inputs = tokenizer(e, max_length=cfg.maxlen, truncation=True)
            input_ids.append(inputs['input_ids'])
            input_mask.append(inputs['attention_mask'])
        input_ids = torch.tensor(sequence_padding(input_ids)).long()
        input_mask = torch.tensor(sequence_padding(input_mask)).float()
        return input_ids, input_mask

    all_e1_vecs = []
    all_e2_vecs = []
    all_labels = []
    for line in data:
        e1, e2, label = line['e1'], line['e2'], line['label']
        e1, e2 = re_match(e1), re_match(e2)
        input_ids, input_mask = convert_to_ids([e1, e2])
        label = torch.tensor([label], dtype=torch.float)
        model.eval()
        with torch.no_grad():
            input_ids, input_mask = input_ids.to(cfg.device), input_mask.to(cfg.device)
            output,_= model(input_ids, input_mask)
            all_e1_vecs.append(output[0].cpu().numpy())
            all_e2_vecs.append(output[1].cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_e1_vecs = np.array(all_e1_vecs)
    all_e2_vecs = np.array(all_e2_vecs)
    all_labels = np.array(all_labels)

    e1_vecs = l2_normalize(all_e1_vecs)
    e2_vecs = l2_normalize(all_e2_vecs)
    sims = (e1_vecs*e2_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(all_labels, sims)
    print('corrcoef', corrcoef)
    return corrcoef





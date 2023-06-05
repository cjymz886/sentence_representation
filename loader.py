import json
import numpy as np
import torch
import re




def re_match(text):
    """bert中词表没有中文类的引号，替换成英文，减少[UNK]"""
    text = text.replace('“','"')
    text = text.replace('”','"')
    text = text.replace("‘","'")
    text = text.replace("’", "'")
    text = re.sub('\s+', '', text)
    text = text.lower()
    return text


def load_lcqmc(filename):
    D = []
    maxlen = 0
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            assert len(line) == 3
            e1, e2, label = line
            e_len = max([len(e1), len(e2)])
            if e_len > maxlen:
                maxlen = e_len
            D.append({'e1':e1, 'e2':e2, 'label':int(label)})
    print('maxlen',maxlen)
    return D



def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


class Collator(object):
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
    def __call__(self, batch):
        batch_token, batch_mask , batch_label = [], [], []
        for line in batch:
            e1, e2, label = line['e1'], line['e2'], line['label']
            e1, e2 = re_match(e1), re_match(e2)
            for e in [e1, e2]:
                inputs = self.tokenizer(e, max_length=self.cfg.maxlen, truncation=True)
                batch_token.append(inputs['input_ids'])
                batch_mask.append(inputs['attention_mask'])
                batch_label.append(label)
        batch_token = torch.tensor(sequence_padding(batch_token)).long()
        batch_mask = torch.tensor(sequence_padding(batch_mask)).float()
        batch_label = torch.tensor(np.array(batch_label)).float()
        return batch_token, batch_mask, batch_label

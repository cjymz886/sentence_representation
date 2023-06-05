import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel





class SentenceRepresenation(BertPreTrainedModel):
    def __init__(self,config):
        super(SentenceRepresenation,self).__init__(config)
        self.bert = BertModel(config=config)

    def forward(self, batch_token, batch_mask):
        embed, neg_embed = self.get_embed(batch_token, batch_mask)
        return embed, neg_embed

    def get_embed(self,token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long(),output_hidden_states=True)
        embed = bert_out[0][:,0]
        neg_embed = bert_out.hidden_states[-2][:,0]     #获取最后第二层embedding作为负样本
        return embed, neg_embed



class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = 0.7
        self.emb_name= 'embeddings.word_embeddings'
        self.alpha = 0.3

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]

        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data  - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return  self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]



class EMA(object):

    def __init__(self,
                 parameters,
                 decay,
                 use_num_updates=True):

        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates +=1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay*(s_param-param))
    def copy_to(self, parameters):
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters if param.requires_grad]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            if param.requires_grad:
                param.data.copy_(c_param.data)
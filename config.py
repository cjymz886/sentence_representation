import torch

class Config():
    maxlen = 64
    random_seed = 2009
    weight_decay=0.0
    max_grad_norm=1.0
    warmup=0.0
    min_num=1e-8
    batch_size = 64
    epochs = 5
    lr = 2e-5
    min_lr = 1e-7
    device = torch.device("cuda")

    train_path = r'E:\open_data\lcqmc\lcqmc\train.tsv'
    dev_path = r'E:\open_data\lcqmc\lcqmc\dev.tsv'
    test_path = r'E:\open_data\lcqmc\lcqmc\test.tsv'
    save_path = './save_models/best-model.bin'

    bert_path=r'E:\pretraing_models\torch\RoBERTa_zh_L12_PyTorch'
    config_path = bert_path + r'\bert_config.json'
    checkpoint_path = bert_path + r'\pytorch_model.bin'
    dict_path = bert_path + r'\vocab.txt'
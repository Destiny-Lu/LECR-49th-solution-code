# %%
import pandas as pd
import os 
import json 
import torch
import torch.nn as nn
import numpy as np 
import logging
import math
import time
import random
import torch.nn.functional as F 

from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CosineSimilarity, CosineEmbeddingLoss, MSELoss
from torch.nn.functional import cosine_similarity, mse_loss
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
os.environ['CUDA_VISIBLE_DEVICES']='7'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class CFG:
    version = '98359_rdrop'
    print_freq = 100
    num_workers = 4
    model = '/home/yxlu/LearningEquality/retriever_models/paraphrase-multilingual-mpnet-base-v2-exp_fold0_epochs20'
    tokenizer = AutoTokenizer.from_pretrained(model)
    gradient_checkpointing = False
    warmup_ratio = 0.1
    epochs = 6
    lr = 1e-5
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 8
    weight_decay = 0.01
    max_grad_norm = 0.012
    seed = 42
    rdrop = True
    rdrop_alpha = 4
    listwise_loss = False 
    cos_loss = True 
    

logger = logging.getLogger('destinylu')
device = torch.device('cuda')
MID_SEP = [CFG.tokenizer.sep_token_id] if CFG.tokenizer.sep_token != '</s>' else [CFG.tokenizer.sep_token_id] * 2
PAD = CFG.tokenizer.pad_token_id


currentTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def setLogger(logger):
# 设置日志级别
    logger.setLevel(logging.DEBUG)
    if not os.path.exists('rerank_log'):
        os.mkdir('rerank_log')

    # 创建文件处理器，将日志输出到文件中
    file_handler = logging.FileHandler('rerank_log/' + currentTime + ' ' + CFG.version)

    # 创建控制台处理器，将日志输出到控制台
    console_handler = logging.StreamHandler()

    # 设置日志格式化器
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

setLogger(logger)

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings  

class CustomDataset(Dataset):
    def __init__(self, df, phase='train'):
        self.data = df 
        self.phase = phase

    def __len__(self):
        return len(self.data)
    
    def pad_and_mask(self, input_ids):
        max_len = max([len(x) for x in input_ids])
        padded_input_ids = []
        padded_attention_mask = []
        for input_ in input_ids:
            attention_mask = [1 for _ in range(len(input_))]
            input_.extend([PAD] * (max_len - len(input_)))
            padded_input_ids.append(input_)
            attention_mask.extend([0] * (max_len - len(attention_mask)))
            padded_attention_mask.append(attention_mask)
            assert len(padded_attention_mask) == len(padded_input_ids)

        return padded_input_ids, padded_attention_mask

    def __getitem__(self, idx):
        data_idx = json.loads(self.data[idx].strip())

        for k in data_idx.keys():
            topic_input_ids = json.loads(k)
        assert len(data_idx.keys()) == 1
        input_ids = []
        labels = []
        for input_ in data_idx[k]['pos']:
            input_ids.append(json.loads(input_))
            labels.append(1.)
        for input_ in data_idx[k]['neg']:
            input_ids.append(json.loads(input_))
            labels.append(-1.)
        if self.phase == 'train':
            combined = list(zip(input_ids, labels))
            random.shuffle(combined)
            input_ids[:], labels[:] = zip(*combined)

        input_ids.append(topic_input_ids)
        padded_input_ids, padded_attention_mask = self.pad_and_mask(input_ids)
        assert len(labels) == len(padded_attention_mask) - 1

        return torch.tensor(padded_input_ids).long(), torch.tensor(padded_attention_mask), torch.tensor(labels)

class uns_model(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.config = AutoConfig.from_pretrained(CFG.model)
        self.model = AutoModel.from_pretrained(CFG.model, config = self.config)
        self.pool = MeanPooling()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, attention_mask)
        
        return feature
class custom_loss(nn.Module):
    def __init__(self) -> None:
        super(custom_loss, self).__init__()
        self.cosloss = nn.CosineEmbeddingLoss()
    
    def listwise_loss(self, topic, content, labels):
        print(topic.shape, content.shape)
        scores  = torch.mm(topic, content.T)
        labels = torch.tensor([torch.tensor(4.).float() if l == 1 else torch.tensor(-1.).float() for l in labels], requires_grad=True).float().to(device)
        labels = F.softmax(labels)
        scores = F.log_softmax(scores)
        loss = F.kl_div(scores, labels, reduction='batchmean')
        
        return loss 

    def forward(self, topic, content, labels):
        loss = 0.
        assert CFG.listwise_loss or CFG.cos_loss
        if CFG.listwise_loss:
            listwise_loss = self.listwise_loss(topic, content, labels)
            loss += listwise_loss
        if CFG.cos_loss:
            loss += self.cosloss(topic.repeat(len(content), 1), content, labels)
            
        return loss

# %%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

# %%
# =========================================================================================
# Train function loop
# =========================================================================================
def train_fn(trainloader, model, criterion, optimizer, epoch, scheduler, device, CFG):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (input_ids, attention_mask, targets) in enumerate(trainloader):
        input_ids = input_ids.squeeze().to(device)
        attention_mask = attention_mask.squeeze().to(device)
        targets = targets.squeeze().to(device)

        with torch.cuda.amp.autocast(enabled = True):
            embedings = model(input_ids, attention_mask)

        topic_emb = embedings[-1, :].unsqueeze(0)
        content_emb = embedings[:-1, :]
        loss = criterion(topic_emb, content_emb, targets)
        losses.update(loss.item(), CFG.batch_size)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1
        scheduler.step()
        if step % CFG.print_freq == 0 or step == (len(trainloader) - 1):
            logger.info('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch + 1, 
                          step, 
                          len(trainloader), 
                          remain = timeSince(start, float(step + 1) / len(trainloader)),
                          loss = losses,
                          grad_norm = grad_norm,
                          lr = scheduler.get_lr()[0]))
            
    return losses.avg

# %%
def valid_fn(valid_loader, model, criterion, device, cfg):
    losses = AverageMeter()
    model.eval()
    start = end = time.time()
    for step, (input_ids, attention_mask, targets) in enumerate(valid_loader):
        input_ids = input_ids.squeeze().to(device)
        attention_mask = attention_mask.squeeze().to(device)
        targets = targets.squeeze().to(device)

        with torch.cuda.amp.autocast(enabled = True):
            embedings = model(input_ids, attention_mask)

        topic_emb = embedings[-1, :].unsqueeze(0)
        content_emb = embedings[:-1, :]
        loss = criterion(topic_emb, content_emb, targets)
        losses.update(loss.item(), cfg.batch_size)
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader) - 1):
            logger.info('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, 
                          len(valid_loader),
                          loss = losses,
                          remain = timeSince(start, float(step + 1) / len(valid_loader))))

    return losses.avg

# %%
train_data = open('./all_listwise.data', 'r').readlines() 
test_data = open('./all_listwise_test.data', 'r').readlines() 

test_data[0]
data_idx = json.loads(test_data[0].strip())
train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data, 'test')
trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=4)
testloader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4)

# %%
model = uns_model(CFG)
model = model.to(device)

criterion = custom_loss()
def get_optimizer_params(model, lr, weight_decay = 0.0):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'lr': lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'lr': lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
        'lr': lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters
optimizer_parameters = get_optimizer_params(
    model, 
    lr = CFG.lr, 
    weight_decay = CFG.weight_decay
)
optimizer = AdamW(
    optimizer_parameters, 
    lr = CFG.lr, 
    eps = CFG.eps, 
    betas = CFG.betas
)
num_train_steps = int(len(trainloader) * CFG.epochs)
num_warmup_steps = num_train_steps * CFG.warmup_ratio
# Scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps = num_warmup_steps, 
    num_training_steps = num_train_steps, 
)

# %%
best_score = 9999999
for epoch in range(CFG.epochs):

    start_time = time.time()
    # Train
    avg_loss = train_fn(trainloader, model, criterion, optimizer, epoch, scheduler, device, CFG)
    # Validation
    avg_val_loss = valid_fn(testloader, model, criterion, device, CFG)
    # Compute f2_score
    # score, threshold = get_best_threshold(x_val, predictions, correlations)
    elapsed = time.time() - start_time
    logger.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
    logger.info(f'Epoch {epoch+1} - test_loss: {avg_val_loss:.4f}')
    if avg_val_loss < best_score:
        best_score = avg_val_loss
        logger.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
        model.save_pretrained(f"{best_score:.4f}_" + CFG.version)

# %%
avg_val_loss = valid_fn(testloader, model, criterion, device, CFG)



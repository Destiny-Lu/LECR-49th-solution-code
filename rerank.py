# %%
# =========================================================================================
# Libraries
# =========================================================================================
import os

import gc
import time
import math
import logging
import random
import warnings
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
from datetime import datetime
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import tokenizers
import transformers
import torch
from torch.optim.optimizer import Optimizer
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from sklearn.model_selection import StratifiedGroupKFold

os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TOKENIZERS_PARALLELISM']='true'
warnings.filterwarnings("ignore")
logger = logging.getLogger('destinylu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================================================
# Configurations
# =========================================================================================
class CFG:
    # model = './stage_one_models/add_context_epochs20'
    model = './all-MiniLM-L6-v2'
    # rerank_data = 'train_top50_pos_0.84226.csv'
    rerank_data = './train_top50_pos_0.84226.csv'
    data_file = './data/correlations.csv'
    
    version = '84224_rdrop_latedrop*'
    
    simcse = False
    fgm = False
    rdrop = True
    late_drop_out = False
    
    print_freq = 500
    num_workers = 4
    tokenizer = AutoTokenizer.from_pretrained(model)
    gradient_checkpointing = False
    num_cycles = 0.5
    warmup_ratio = 0.1
    epochs = 6
    encoder_lr = 2e-5
    decoder_lr = 1e-4
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 64
    weight_decay = 0.01
    max_grad_norm = 0.012
    max_len = 256
    n_folds = 5
    seed = 42
    rdrop_alpha = 4

MIDSEP = [CFG.tokenizer.sep_token_id] if CFG.tokenizer.sep_token != '</s>' else [CFG.tokenizer.sep_token_id] * 2
PADID = CFG.tokenizer.pad_token_id
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

# =========================================================================================
# Seed everything for deterministic results
# =========================================================================================
def seed_everything(cfg):
    random.seed(cfg.seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True     
    
# =========================================================================================
# F2 score metric
# =========================================================================================
def f2_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)


# =========================================================================================
# Get max length
# =========================================================================================
# def get_max_length(train, cfg):
#     lengths = []
#     for text in train['text'].fillna("").values:
#         length = len(cfg.tokenizer(text, add_special_tokens = False)['input_ids'])
#         lengths.append(length)
#     cfg.max_len = max(lengths, 510) + 2 # cls & sep
#     logger.info(f"max_len: {cfg.max_len}")
    

class CustomDataset(Dataset):
    def __init__(self, df):
        topics_info = list(df['topics_info'].values)
        content_info = list(df['content_info'].values)
        targets = list(df['target'].values)
        data = []
        for topic, content, target in zip(topics_info, content_info, targets):
            data.append((topic, content, target))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1, sentence2, target = self.data[idx]
        encoded = CFG.tokenizer.encode_plus(
            sentence1, sentence2,
            add_special_tokens=True,
            max_length=128,
            truncation_strategy='longest_first',
            padding='max_length',
            return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), torch.tensor(target)

    
# =========================================================================================
# Collate function for training
# =========================================================================================
def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs

# =========================================================================================
# Mean pooling class
# =========================================================================================
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

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim=384):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.att_fc2 = nn.Linear(hidden_dim, 1)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
                
    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  
        return x
    
# =========================================================================================
# Model
# =========================================================================================
class SimCSE(nn.Module):
    def __init__(self, scale=20.) -> None:
        super(SimCSE, self).__init__()
        self.scale = scale

    # '''
    # def simcse_loss(y_true, y_pred):
    #     idxs = K.arange(0, K.shape(y_pred)[0])
    #     idxs_1 = idxs[None, :]
    #     idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    #     y_true = K.equal(idxs_1, idxs_2)
    #     y_true = K.cast(y_true, K.floatx())
    #     y_pred = K.l2_normalize(y_pred, axis=1)
    #     similarities = K.dot(y_pred, K.transpose(y_pred))
    #     similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    #     similarities = similarities * 20
    #     loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    #     return K.mean(loss)
    
    # translate to torch by ChatGPT:
    
    # def simcse_loss(y_true, y_pred):
    #     idxs = torch.arange(0, y_pred.shape[0])
    #     idxs_1 = idxs[None, :]
    #     idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    #     y_true = torch.eq(idxs_1, idxs_2)
    #     y_true = y_true.float()
    #     y_pred = torch.nn.functional.normalize(y_pred, dim=1)
    #     similarities = torch.mm(y_pred, y_pred.transpose(0, 1))
    #     similarities = similarities - torch.eye(y_pred.shape[0]) * 1e12
    #     similarities = similarities * 20
    #     loss = torch.nn.functional.cross_entropy(similarities, y_true.long(), reduction='mean')
    #     return loss
    # '''
    def forward(self, feature_1, feature_2):
        bsz = feature_1.shape[0] 
        labels = torch.zeros((bsz*2, bsz*2), dtype=torch.float).to(device)
        '''
        labels just like: 
        [[0, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 0]]
        '''
        for p in range(0, bsz*2, 2):
            labels[p][p//2+bsz] = 1
            labels[p+1][p//2] = 1
        feature = torch.concat((feature_1, feature_2), dim=0)  # (bsz*2, emb_size)
        feature = F.normalize(feature, dim=-1)
        sim_matrix = torch.mm(feature, feature.transpose(0, 1))
        sim_matrix = sim_matrix - torch.eye(sim_matrix.shape[0]).to(device) * 1e12
        sim_matrix = sim_matrix * 20
        loss = F.cross_entropy(sim_matrix, labels, reduction='mean')
        # logger.info("simcse loss: %f" % loss)
        return loss * 0.02

class custom_model(nn.Module):
    def __init__(self, cfg):
        super(custom_model, self).__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states = True)
        # self.config.hidden_dropout = 0.0
        # self.config.hidden_dropout_prob = 0.0
        # self.config.attention_dropout = 0.0
        # self.config.attention_probs_dropout_prob = 0.0
        self.model = AutoModel.from_pretrained(cfg.model, config = self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = AttentionPooling(hidden_dim=self.config.hidden_size)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
        self.simcse = SimCSE()
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def feature(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, attention_mask)
        return feature
    
    def forward(self, input_ids, attention_mask, labels, criterion=None):
        # print(input_ids.shape, attention_mask.shape)
        feature = self.feature(input_ids, attention_mask)
        logits = self.fc(feature)
        loss = 0.
        if criterion==None:
            return logits
        if self.cfg.simcse or self.cfg.rdrop:
            feature_2 = self.feature(input_ids, attention_mask)
            logits_2 = self.fc(feature_2)
            # print(logits, labels)
            loss = (criterion(logits.view(-1), labels.float()) + criterion(logits_2.view(-1), labels.float())) / 2
            if self.cfg.simcse:
                loss += self.simcse(feature, feature_2)
            if self.cfg.rdrop:
                kl_loss1 = F.kl_div(F.log_softmax(feature, dim=-1), F.softmax(feature_2, dim=-1), reduce=None)
                kl_loss2 = F.kl_div(F.log_softmax(feature_2, dim=-1), F.softmax(feature, dim=-1), reduce=None)
                kl_loss = (kl_loss1 + kl_loss2) / 2
                loss += self.cfg.rdrop_alpha * kl_loss
        else:
            loss = criterion(logits.view(-1), labels.float())


        return loss
    
# =========================================================================================
# Helper functions
# =========================================================================================
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
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# =========================================================================================
# Train function loop
# =========================================================================================
def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    if cfg.fgm:
        fgm = FGM(model)
    for step, (input_ids, attention_mask, targets) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast(enabled = True):
            loss = model(input_ids, attention_mask, targets, criterion)
        losses.update(loss.item(), cfg.batch_size)
        scaler.scale(loss).backward()
        if cfg.fgm:
            fgm.attack()
            with torch.cuda.amp.autocast(enabled = True):
                loss_sum = model(input_ids, attention_mask, targets, criterion)
            scaler.scale(loss_sum).backward()
            fgm.restore()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1
        scheduler.step()
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):
            logger.info('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch + 1, 
                          step, 
                          len(train_loader), 
                          remain = timeSince(start, float(step + 1) / len(train_loader)),
                          loss = losses,
                          grad_norm = grad_norm,
                          lr = scheduler.get_lr()[0]))
    return losses.avg



# %%
# =========================================================================================
# Valid function loop
# =========================================================================================
def valid_fn(valid_loader, model, criterion, device, cfg):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (input_ids, attention_mask, targets) in enumerate(valid_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            y_preds = model(input_ids, attention_mask, targets)
        loss = criterion(y_preds.view(-1), targets.float())
        losses.update(loss.item(), cfg.batch_size)
        preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader) - 1):
            logger.info('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, 
                          len(valid_loader),
                          loss = losses,
                          remain = timeSince(start, float(step + 1) / len(valid_loader))))
    predictions = np.concatenate(preds, axis = 0)
    return losses.avg, predictions

# =========================================================================================
# Get best threshold
# =========================================================================================
def get_best_threshold(x_val, val_predictions, correlations):
    best_score = 0
    best_threshold = None
    for thres in np.arange(0.01, 1, 0.01):
        x_val['predictions'] = np.where(val_predictions > thres, 1, 0)
        x_val1 = x_val[x_val['predictions'] == 1]
        x_val1 = x_val1.groupby(['topics_ids'])['content_ids'].unique().reset_index()
        x_val1['content_ids'] = x_val1['content_ids'].apply(lambda x: ' '.join(x))
        x_val1.columns = ['topic_id', 'predictions']
        x_val0 = pd.Series(x_val['topics_ids'].unique())
        x_val0 = x_val0[~x_val0.isin(x_val1['topic_id'])]
        x_val0 = pd.DataFrame({'topic_id': x_val0.values, 'predictions': ""})
        x_val_r = pd.concat([x_val1, x_val0], axis = 0, ignore_index = True)
        x_val_r = x_val_r.merge(correlations, how = 'left', on = 'topic_id')
        score = f2_score(x_val_r['content_ids'], x_val_r['predictions'])
        if score > best_score:
            best_score = score
            best_threshold = thres
    return best_score, best_threshold
    
# =========================================================================================
# Train & Evaluate
# =========================================================================================
def train_and_evaluate_one_fold(train, correlations, fold, cfg):
    logger.info(' ')
    logger.info(f"========== fold: {fold} training ==========")
    # Split train & validation
    x_train = train[train['fold'] != fold]
    x_val = train[train['fold'] == fold]
    train_datasets = CustomDataset(x_train)
    train_loader = DataLoader(train_datasets, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_datasets = CustomDataset(x_val)
    valid_loader = DataLoader(valid_datasets, batch_size=cfg.batch_size, shuffle=False, num_workers=4, drop_last=False)
    # Get model
    model = custom_model(cfg)
    model.to(device)
    # Optimizer
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay = 0.0):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
            'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters
    optimizer_parameters = get_optimizer_params(
        model, 
        encoder_lr = cfg.encoder_lr, 
        decoder_lr = cfg.decoder_lr,
        weight_decay = cfg.weight_decay
    )
    optimizer = AdamW(
        optimizer_parameters, 
        lr = cfg.encoder_lr, 
        eps = cfg.eps
    )
    num_train_steps = int(len(x_train) / cfg.batch_size * cfg.epochs)
    num_warmup_steps = num_train_steps * cfg.warmup_ratio
    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps, 
        num_training_steps = num_train_steps, 
        num_cycles = cfg.num_cycles
        )
    # Training & Validation loop
    criterion = nn.BCEWithLogitsLoss(reduction = "mean")
    best_score = 0
    
    if cfg.late_drop_out:
        init_drop_out_record = {}
    
    for epoch in range(cfg.epochs):
        if cfg.late_drop_out:
            if epoch == 0:
                for name, module in model.named_modules():
                    if isinstance(module, nn.Dropout):
                        init_drop_out_record[name] = module.p
                        module.p = 0
            elif epoch == 3:
                for name, module in model.named_modules():
                    if isinstance(module, nn.Dropout):
                        module.p = init_drop_out_record[name]
                
        start_time = time.time()
        # Train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg)
        # Validation
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device, cfg)
        # Compute f2_score
        score, threshold = get_best_threshold(x_val, predictions, correlations)
        elapsed = time.time() - start_time
        logger.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        logger.info(f'Epoch {epoch+1} - Score: {score:.4f} - Threshold: {threshold:.5f}')
        if score > best_score:
            best_score = score
            logger.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(
                {'model': model.state_dict(), 'predictions': predictions}, 
                f"fold{fold}_{cfg.seed}_ver5_{best_score:.4f}_" + cfg.version + '.pth'
                )
            val_predictions = predictions
    torch.cuda.empty_cache()
    gc.collect()
    # Get best threshold
    best_score, best_threshold = get_best_threshold(x_val, val_predictions, correlations)
    logger.info(f'Our CV score is {best_score} using a threshold of {best_threshold}')



# %%
# Seed everything
seed_everything(CFG)
# Read data
train = pd.read_csv(CFG.rerank_data)
# correlations = pd.read_csv('./data/correlations.csv')
correlations = pd.read_csv(CFG.data_file)
# Get max length
# get_max_length(train, CFG)
# Train and evaluate one fold


# %%
train_and_evaluate_one_fold(train, correlations, 0, CFG)



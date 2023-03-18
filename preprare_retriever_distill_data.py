# %%
import os 
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import csv

from typing import *
from tqdm import tqdm
from pathlib import Path as path
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from transformers import DataCollatorWithPadding
from utils import prepare_topic_text, prepare_content_text, Topic, adapt_text, count_words
from transformers import AutoTokenizer, AutoConfig, AutoModel
os.environ['CUDA_VISIBLE_DEVICES']='5'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DATA_PATH = './data/'
pretrained_model_type = 'stage_one_models/all-mini'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_type)
device = torch.device('cuda')

# %%
def read_data(datapath, tokenizer):

    topics_df = pd.read_csv(datapath + 'topics.csv', index_col=0).fillna({"title": "", "description": ""})
    topics = pd.read_csv(datapath + 'topics.csv')
    mask = topics['has_content'] != False
    topics = topics[mask]
    topics.fillna('', inplace = True)
    
    topic_info = []
    for topic_id, lang, title, description in tqdm(zip(topics['id'].values, topics['language'].values, topics['title'].values, topics['description'].values), total=len(topics['title'])):
        topic_context = Topic(topic_id, topics_df)
        # print(topic_context.title)
        tree = topic_context.get_breadcrumbs()
        # tree_title = [a.title for a in tree]
        depth = len(tree)
        if depth == 1:
            context_title, context_description = '', ''
        else:
            context_title = adapt_text(tree[1].title, lang, 24)
            context_description = adapt_text(tree[1].description, lang, 48 - count_words(context_title, lang))
        text = prepare_topic_text(lang, depth, title, description, f'{context_title} {context_description}')
        text = tokenizer(text, truncation=True, max_length=128, add_special_tokens=True)['input_ids']
        topic_info.append(text)

    topics['info'] = topic_info
    topics.drop(['channel', 'category', 'level', 'parent', 'has_content', 'title', 'description', 'language'], axis = 1, inplace = True)
    del topic_info, topics_df
    gc.collect()

    content = pd.read_csv(datapath + "content.csv")
    content.fillna('', inplace = True)
    content_info = []
    for lang, title, description, text in tqdm(zip(content['language'].values, content['title'].values, content['description'].values, content['text'].values), total=len(content['title'])):
        text = prepare_content_text(lang, title, description, text)
        text = tokenizer(text, truncation=True, max_length=128, add_special_tokens=True)['input_ids']
        content_info.append(text)
        
    
    content['info'] = content_info
    
    content.drop(['kind', 'copyright_holder', 'license', 'title', 'description', 'language', 'text'], axis = 1, inplace = True)
    del content_info
    gc.collect()

    correlations = pd.read_csv(DATA_PATH + "10folds_correlations.csv")
    # kfolds = cv_split(correlations, 10, 42)
    correlations = correlations[correlations.fold != 0]

    topics.reset_index(drop = True, inplace = True)
    content.reset_index(drop = True, inplace = True)

    topics.rename(columns=lambda x: "topic_" + x, inplace=True)
    content.rename(columns=lambda x: "content_" + x, inplace=True)

    correlations["content_id"] = correlations["content_ids"].str.split(" ")
    corr = correlations.explode("content_id").drop(columns=["content_ids"])

    corr = corr.merge(topics, how="left", on="topic_id")
    corr = corr.merge(content, how="left", on="content_id")

    return corr, correlations

# %%
corr, kflods = read_data(DATA_PATH, tokenizer)

# %%
torch.tensor(corr['topic_info'][0])

# %%
def get_embeddings(loader, model, device):
    model.eval()
    preds = []
    for step, inputs in enumerate(tqdm(loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
            # print(y_preds)
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    return preds

# %%
class uns_dataset(Dataset):
    def __init__(self, df, type_):
        if type_ == 'topics':
            self.input_ids = df['topic_info']
        else:
            self.input_ids = df['content_info']
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, item):
        input_ids = self.input_ids[item]
        print(input_ids, '^^^^^')
        attention_mask = [1 for _ in range(len(input_ids))]
        print(attention_mask)
        return {'input_ids':input_ids, 'attention_mask': attention_mask}

# %%
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

# =========================================================================================
# Unsupervised model
# =========================================================================================
class uns_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_type)
        self.model = AutoModel.from_pretrained(pretrained_model_type, config = self.config)
        self.pool = MeanPooling()
    
    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        
        return feature

# %%
def get_neighbors(topics, content):
    # Create topics dataset
    topics_dataset = uns_dataset(topics, 'topics')
    # Create content dataset
    content_dataset = uns_dataset(content, 'comtent')
    # Create topics and content dataloaders
    topics_loader = DataLoader(
        topics_dataset, 
        batch_size = 32, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = tokenizer, padding = 'longest'),
        num_workers = 4, 
        pin_memory = False, 
        drop_last = False
    )
    content_loader = DataLoader(
        content_dataset, 
        batch_size = 32, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = tokenizer, padding = 'longest'),
        num_workers = 4, 
        pin_memory = False, 
        drop_last = False
        )
    model = uns_model()
    model.to(device)
    # Predict topics
    topics_preds = get_embeddings(topics_loader, model, device)
    content_preds = get_embeddings(content_loader, model, device)
    # Release memory
    torch.cuda.empty_cache()

    return topics_preds, content_preds

# %%
topics_preds, content_preds = get_neighbors(corr, corr)

# %%


# %%




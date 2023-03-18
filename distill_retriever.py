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
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader, BatchSampler, Sampler
from sklearn.model_selection import KFold
from utils import prepare_topic_text, prepare_content_text, Topic, adapt_text, count_words

os.environ['CUDA_VISIBLE_DEVICES']='1'

TEACHER_MODEL_PATH = './pmm-v2'
STUDENT_MODEL_PATH = './pmm-v2'

DATA_PATH = "./data/"
MODEL_OUTPUT_PATH = './model_output'
CHECKPOINT_PATH = './checkpoint'
CORR_PATH = './saved_@_corr.csv'
# KFOLDS_PATH = './saved_@_kfolds.csv'

device = 'cuda'
use_tqdm = True
save_corr = True
rewrite_saved_data = False

batch_size = 32
num_epochs = 10


teacher_corr_path = CORR_PATH.replace('@', TEACHER_MODEL_PATH.split('/')[-1])
student_corr_path = CORR_PATH.replace('@', STUDENT_MODEL_PATH.split('/')[-1])
if rewrite_saved_data:
    try:
        os.remove(teacher_corr_path)
        os.remove(student_corr_path)
    except FileNotFoundError:
        pass
    
def cv_split(train, n_folds, seed):
    kfold = KFold(n_splits = n_folds, shuffle = True, random_state = seed)
    for num, (train_index, val_index) in enumerate(kfold.split(train)):
        train.loc[val_index, 'fold'] = int(num)
    train['fold'] = train['fold'].astype(int)
    return train

def read_data(datapath, tokenizer, saved_path):
    # >>> read saved data
    if path(saved_path).exists():
        return pd.read_csv(saved_path), None
    # >>>
    topics_df = pd.read_csv(datapath + 'topics.csv', index_col=0).fillna({"title": "", "description": ""})
    topics = pd.read_csv(datapath + 'topics.csv')
    mask = topics['has_content'] != False
    topics = topics[mask]
    topics.fillna('', inplace = True)
    
    topic_info = []
    
    # >>> use tqdm
    iter_target = zip(topics['id'].values, topics['language'].values, topics['title'].values, topics['description'].values)
    if use_tqdm:
        iter_target = tqdm(iter_target, total=len(topics['title']))
    # >>>
    
    for topic_id, lang, title, description in iter_target:
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
        text = tokenizer(text, truncation=True, max_length=128, add_special_tokens=False)['input_ids']
        text = tokenizer.convert_ids_to_tokens(text)
        topic_info.append(''.join(text))

    topics['info'] = topic_info
    topics.drop(['channel', 'category', 'level', 'parent', 'has_content', 'title', 'description', 'language'], axis = 1, inplace = True)
    del topic_info, topics_df
    gc.collect()

    content = pd.read_csv(datapath + "content.csv")
    content.fillna('', inplace = True)
    content_info = []
    
    # >>> use tqdm
    iter_target = zip(content['language'].values, content['title'].values, content['description'].values, content['text'].values)
    if use_tqdm:
        iter_target = tqdm(iter_target, total=len(content['title']))
    # >>>
    
    for lang, title, description, text in iter_target:
        text = prepare_content_text(lang, title, description, text)
        text = tokenizer(text, truncation=True, max_length=128, add_special_tokens=False)['input_ids']
        text = tokenizer.convert_ids_to_tokens(text)
        content_info.append(''.join(text))
        
    
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

print(f'===== start reading data =====')
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH)
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH)
student_corr, _ = read_data(DATA_PATH, student_tokenizer, student_corr_path)
teacher_corr, _ = read_data(DATA_PATH, teacher_tokenizer, teacher_corr_path)
# >>> save data   
if save_corr:
    if not path(student_corr_path).exists():
        student_corr.to_csv(student_corr_path)
    if not path(teacher_corr_path).exists():
        teacher_corr.to_csv(teacher_corr_path)
# >>>
print(f'===== finish reading data =====')


class CustomBatchSampler(BatchSampler):
    def __init__(self, dataset: List[InputExample], batch_size):
        self.dataset = dataset  # [InputExample(texts=[s1,s2], label=0)]
        self.batch_size = batch_size
    
    def __iter__(self):
        order_list = list(range(len(self.dataset)))
        np.random.shuffle(order_list)
        
        while len(order_list):
            topic_set, content_set = set(), set()
            cur = []
            for p in range(len(order_list)-1, -1, -1):
                ct, cc = self.dataset[order_list[p]].texts
                if ct not in topic_set and cc not in content_set:
                    cur.append(p)
                    topic_set.add(ct)
                    content_set.add(cc)
                    if len(cur) >= self.batch_size:
                        break
            batch_data = [order_list[p]for p in cur]
            for d in cur:
                del order_list[d]
            if len(batch_data) < 2:
                return
            yield batch_data
    
    # TODO: real length is larger than length of dataset / batch_size but smaller than length of dataset
    def __len__(self) -> int:
        return len(self.dataset)

def get_dataloader(train_tuples, n_tuple):
    # 加入description和text
    train_df = pd.DataFrame(train_tuples)

    dataset = Dataset.from_pandas(train_df)
    # device = torch.device('cuda')

    train_examples = []
    # train_data = dataset["set"]
    n_examples = dataset.num_rows

    for i in range(n_examples):
        example = train_tuples[i]
        if example[0] == None:  # remove None
            print(example)
            continue        
        train_examples.append(InputExample(texts=[str(example[p])for p in range(n_tuple)]))

    # batch_sampler = CustomBatchSampler(train_examples, batch_size)
    # train_dataloader = DataLoader(train_examples, batch_sampler=batch_sampler)
    train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)
    
    # model = SentenceTransformer("pmm-v2").to(device)
    # train_dataloader.collate_fn = model.smart_batching_collate
    # p = next(iter(train_dataloader))
    # print(p)
    # exit()
    return train_dataloader


class DistillLoss(nn.Module):
    def __init__(self, teacher_model_, student_model_):
        super().__init__()
        self.teacher_model = teacher_model_
        self.student_model = student_model_
        self.mse = nn.MSELoss()

    def forward(self, sentence_features, labels):
        # print('-'*20)
        # print(sentence_features)
        # print(type(sentence_features))
        # print(len(sentence_features))
        # print('-'*20)
        with torch.no_grad():
            emb_a = self.teacher_model(sentence_features[0])['sentence_embedding']
        emb_b = self.student_model(sentence_features[1])['sentence_embedding']
        return self.mse(emb_a, emb_b)
        # reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        # embeddings_a = reps[0]
        # embeddings_b = torch.cat(reps[1:])

        # scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        # labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        # return self.cross_entropy_loss(scores, labels)


teacher_model = SentenceTransformer(TEACHER_MODEL_PATH)
student_model = SentenceTransformer(STUDENT_MODEL_PATH)
teacher_model.eval()
student_model.train()

train_loss1 = losses.MultipleNegativesRankingLoss(model=student_model)
train_tuples1 = student_corr[['topic_info', 'content_info']].values.tolist()
train_dataloader1 = get_dataloader(train_tuples1, 2)

# below losses are the same
train_loss2 = DistillLoss(teacher_model, student_model)  

topic_corr = pd.concat([teacher_corr['topic_info'], student_corr['topic_info']], axis=1)
# train_tuples2 = topic_corr.values.tolist()
# train_dataloader2 = get_dataloader(train_tuples2, 2)

content_corr = pd.concat([teacher_corr['content_info'], student_corr['content_info']], axis=1)
# train_tuples3 = content_corr.values.tolist()
# train_dataloader3 = get_dataloader(train_tuples3, 2)

tc = topic_corr.rename(columns={topic_corr.columns[0]: '0', topic_corr.columns[1]:'1'})
cc = content_corr.rename(columns={content_corr.columns[0]: '0', content_corr.columns[1]:'1'})
topic_content_corr = pd.concat([tc, cc], axis=0)
train_tuples4 = topic_content_corr.values.tolist()
train_dataloader4 = get_dataloader(train_tuples4, 2)

warmup_steps = int(len(train_dataloader4) * num_epochs * 0.1)  # 10% of train data

student_model.fit(
    train_objectives=[
        # (train_dataloader1, train_loss1),
        # (train_dataloader2, train_loss2),
        # (train_dataloader3, train_loss2),
        (train_dataloader4, train_loss2),
    ],
    epochs=num_epochs,
    save_best_model = True,
    warmup_steps=warmup_steps,
    output_path=MODEL_OUTPUT_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    checkpoint_save_steps=4000,
    use_amp=True,
    # >>> modify optimizer here <<<
    optimizer_class = torch.optim.AdamW,
    optimizer_params = {'lr': 2e-5},
)
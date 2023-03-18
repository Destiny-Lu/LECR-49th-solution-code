# %%
import pandas as pd
import gc
import copy
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from datasets import Dataset
from torch.utils.data import DataLoader, Sampler, default_collate
from sklearn.model_selection import KFold
from utils import prepare_topic_text, prepare_content_text, Topic, adapt_text, count_words
from transformers import AutoTokenizer

import os 
# os.environ['CUDA_VISIBLE_DEVICES']='5'
DATA_PATH = "./data/"
pretrained_model_type = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_type)

def cv_split(train, n_folds, seed):
    kfold = KFold(n_splits = n_folds, shuffle = True, random_state = seed)
    for num, (train_index, val_index) in enumerate(kfold.split(train)):
        train.loc[val_index, 'fold'] = int(num)
    train['fold'] = train['fold'].astype(int)
    return train

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
    for lang, title, description, text in tqdm(zip(content['language'].values, content['title'].values, content['description'].values, content['text'].values), total=len(content['title'])):
        text = prepare_content_text(lang, title, description, text)
        text = tokenizer(text, truncation=True, max_length=128, add_special_tokens=False)['input_ids']
        text = tokenizer.convert_ids_to_tokens(text)
        content_info.append(''.join(text))
        
    
    content['info'] = content_info
    
    content.drop(['kind', 'copyright_holder', 'license', 'title', 'description', 'language', 'text'], axis = 1, inplace = True)
    del content_info
    gc.collect()

    correlations = pd.read_csv(DATA_PATH + "correlations.csv")
    # kfolds = cv_split(correlations, 10, 42)
    # correlations = kfolds[kfolds.fold != 0]

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
# kflods.to_csv('10folds_correlations.csv')

# %%
# 加入description和text
train_pairs = corr[['topic_info', 'content_info']].values.tolist()
train_df = pd.DataFrame(train_pairs)

dataset = Dataset.from_pandas(train_df)
# device = torch.device('cuda')

train_examples = []
# train_data = dataset["set"]
n_examples = dataset.num_rows

for i in range(n_examples):
    example = train_pairs[i]    
    train_examples.append(InputExample(texts=[str(example[0]), str(example[1])]))
# train_examples = train_examples[:50]


# %%
# model = SentenceTransformer(pretrained_model_type)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=512)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
# num_epochs = 10
num_epochs = 80
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          save_best_model = True,
          output_path='./retriever_models/paraphrase-multilingual-mpnet-base-v2-exp_fold0_epochs20',
          warmup_steps=warmup_steps,
        #   checkpoint_path='./',
        #   checkpoint_save_steps=3500,
          use_amp=True)

# %%




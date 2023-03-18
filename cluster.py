# %%
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import gc
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import DataCollatorWithPadding
from utils import prepare_content_text, prepare_topic_text, Topic, count_words, adapt_text
import cupy as cp
from cuml.neighbors import NearestNeighbors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================================================
# Configurations
# =========================================================================================
class CFG:
    num_workers = 4
    model = '/home/yxlu/LearningEquality/all_train_data_retriever'
    stage_tow_model = 'Epoch2-Model.pth'
    batch_size = 32
    seed = 42
    top_n = 150
    tokenizer = AutoTokenizer.from_pretrained(model)
DATA_PATH = './data/'
MID_SEP = [CFG.tokenizer.sep_token_id] * 2 if CFG.tokenizer.sep_token == '</s>' else [CFG.tokenizer.sep_token_id]

# %%
def read_data():

    topics_df = pd.read_csv(DATA_PATH + 'topics.csv', index_col=0).fillna({"title": "", "description": ""})
    topics = pd.read_csv(DATA_PATH + 'topics.csv')
#     topics.drop(topics.index[5000:], inplace=True)
    # sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')
    # topics = topics.merge(sample_submission, how = 'inner', left_on = 'id', right_on = 'topic_id')
    topics.fillna('', inplace = True)
    # topics.drop(topics.index[500:], inplace=True)

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
        topic_info.append(prepare_topic_text(lang, depth, title, description, f'{context_title} {context_description}')) 
    topic_inputs = CFG.tokenizer(topic_info, truncation=True, max_length=128)
    del topic_inputs['attention_mask']
    gc.collect()
    topics['topic_input_ids'] = topic_inputs['input_ids']
    topics['total_length'] = topics['topic_input_ids'].apply(lambda x: len(x))
    topics = topics.sort_values(by='total_length')
    topics.drop(['channel', 'category', 'level', 'parent', 'has_content', 'total_length', 'title', 'description', 'language'], axis = 1, inplace = True)
    del topic_info, topic_inputs, topics_df
    gc.collect()

    content = pd.read_csv(DATA_PATH + "content.csv")
    content.fillna('', inplace = True)
    content_info = []
    for lang, title, description, text in tqdm(zip(content['language'].values, content['title'].values, content['description'].values, content['text'].values), total=len(content['title'])):
        content_info.append(prepare_content_text(lang, title, description, text))
        
    content_inputs = CFG.tokenizer(content_info, truncation=True, max_length=128)
    del content_inputs['attention_mask']
    gc.collect()
    content['content_input_ids'] = content_inputs['input_ids']
    

    content['total_length'] = content['content_input_ids'].apply(lambda x: len(x))
    content = content.sort_values(by='total_length')
    content.drop(['kind', 'copyright_holder', 'license', 'total_length', 'title', 'description', 'language', 'text'], axis = 1, inplace = True)
    del content_info, content_inputs
    gc.collect()

    topics.reset_index(drop = True, inplace = True)
    content.reset_index(drop = True, inplace = True)

    correlations = pd.read_csv(DATA_PATH + 'correlations.csv')
    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    print(f"correlations.shape: {correlations.shape}")
    

    return topics, content, correlations

# %%
topics, content, correlations = read_data()

# %%
class uns_dataset(Dataset):
    def __init__(self, df):
        if 'topic_input_ids' in df.keys():
            self.input_ids = df['topic_input_ids'].values
        else:
            self.input_ids = df['content_input_ids'].values
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, item):
        input_ids = torch.tensor(self.input_ids[item], dtype = torch.long)
        attention_mask = torch.ones_like(input_ids).long()
        return {'input_ids':input_ids, 'attention_mask': attention_mask}

# %%
# =========================================================================================
# Get the amount of positive classes based on the total
# =========================================================================================
def get_pos_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)


def recall_at_k(y_true, y_pred, k=100):
    """
    Compute the recall@k given true labels and predicted labels.
    Args:
        y_true (list of str): True labels.
        y_pred (list of str): Predicted labels.
        k (int): The number of predicted labels to consider.
    Returns:
        The recall@k score.
    """
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    recall = 0
    true_labels = []
    pred_labels = []
    for i in range(n):
        true_labels = set(y_true[i].split())
        pred_labels = set(y_pred[i][:k].split())
        tp = len(true_labels.intersection(pred_labels))
        recall += tp / len(true_labels)
    recall /= n
    return round(recall, 5)
# =========================================================================================
# F2 Score 
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

# %%
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

# =========================================================================================
# Unsupervised model
# =========================================================================================
class uns_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = AutoConfig.from_pretrained(cfg.model)
        self.model = AutoModel.from_pretrained(cfg.model, config = self.config)
        self.pool = MeanPooling()
    
    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        
        return feature

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
def get_neighbors(topics, content, CFG):
    # Create topics dataset
    topics_dataset = uns_dataset(topics)
    # Create content dataset
    content_dataset = uns_dataset(content)
    # Create topics and content dataloaders
    topics_loader = DataLoader(
        topics_dataset, 
        batch_size = CFG.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = CFG.tokenizer, padding = 'longest'),
        num_workers = CFG.num_workers, 
        pin_memory = False, 
        drop_last = False
    )
    content_loader = DataLoader(
        content_dataset, 
        batch_size = CFG.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = CFG.tokenizer, padding = 'longest'),
        num_workers = CFG.num_workers, 
        pin_memory = False, 
        drop_last = False
        )
    model = uns_model(CFG)
    model.load_state_dict(torch.load(CFG.stage_tow_model))
    model.to(device)
    # Predict topics
    topics_preds = get_embeddings(topics_loader, model, device)
    topics_preds_gpu = cp.array(topics_preds)
    content_preds = get_embeddings(content_loader, model, device)
    content_preds_gpu = cp.array(content_preds)
    # Release memory
    torch.cuda.empty_cache()
    del topics_dataset, content_dataset, topics_loader, content_loader, topics_preds, content_preds
    # KNN model
    print(' ')
    print('Training KNN model...')
    neighbors_model = NearestNeighbors(n_neighbors = CFG.top_n, metric = 'cosine')
    neighbors_model.fit(content_preds_gpu)
    indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance = False)

    
    
    predictions = []
    for k in tqdm(range(len(indices))):
        pred = indices[k]
        p = ' '.join([content.loc[ind, 'id'] for ind in pred.get()])
        predictions.append(p)
    # topics['predictions'] = predictions
    topics['predictions'] = predictions
    del topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices, model
    gc.collect()
    
    return topics

# %%
def build_training_set(topics, content):
    # Create lists for training
    topics_ids = []
    content_ids = []
    topics_info = []
    content_info = []
    targets = []
    folds = []
    # Iterate over each topic
    for k in tqdm(range(len(topics))):
        row = topics.iloc[k]
        topics_id = row['id']
        info1 = row['topic_input_ids']
        predictions = row['predictions'].split(' ')
        ground_truth = set(row['content_ids'].split(' '))
        fold = row['fold']
        
        for pred in predictions:
            topics_ids.append(topics_id)
            content_ids.append(pred)
            info2 = content.loc[pred, 'content_input_ids']
            
            topics_info.append(info1)
            content_info.append(info2)
            folds.append(fold)
            # If pred is in ground truth, 1 else 0
            if pred in ground_truth:
                targets.append(1)
                ground_truth.remove(pred)
            else:
                targets.append(0)
                
        for rest in ground_truth:
            topics_ids.append(topics_id)
            content_ids.append(rest)
            info2 = content.loc[rest, 'content_input_ids']
            
            topics_info.append(info1)
            content_info.append(info2)
            folds.append(fold)
            targets.append(1)

    # Build training dataset
    train = pd.DataFrame(
        {'topics_ids': topics_ids, 
         'content_ids': content_ids, 
         'topics_info': topics_info, 
         'content_info': content_info, 
         'target': targets,
         'fold' : folds}
    )
    # Release memory
    del topics_ids, content_ids, targets
    gc.collect()
    return train

# %%
topics = get_neighbors(topics, content, CFG)
torch.cuda.empty_cache()
topics_test = topics.merge(correlations, how = 'inner', left_on = ['id'], right_on = ['topic_id'])

# %%
pos_score = get_pos_score(topics_test['content_ids'], topics_test['predictions'])
recall_value = recall_at_k(topics_test['content_ids'], topics_test['predictions'], k=CFG.top_n)
print(f'Our max positive score is {pos_score}')
print(f'Ourrecall@{CFG.top_n} score is {recall_value}')
f_score = f2_score(topics_test['content_ids'], topics_test['predictions'])
print(f'Our f2_score is {f_score}')

# %%
content.set_index('id', inplace = True)
full_correlations = pd.read_csv('./data/10folds_correlations.csv')
topics_full = topics.merge(full_correlations, how = 'inner', left_on = ['id'], right_on = ['topic_id'])
topics_full['predictions'] = topics_full.apply(lambda x: ' '.join(list(set(x.predictions.split(' ') + x.content_ids.split(' ')))) \
                                               if x.fold != 0 else x.predictions, axis = 1)

# %%
for content_ids in topics_full['content_ids']:
    if len(set(content_ids.split(' '))) == 0:
        print(1)
        break

# %%
train = build_training_set(topics_full, content)
print(f'Our training set has {len(train)} rows')
train.head()

# %%
# Save train set to disk to train on another notebook
train.to_csv(f'{CFG.model}_train_top{CFG.top_n}_pos_{pos_score}.csv', index = False)

# %%




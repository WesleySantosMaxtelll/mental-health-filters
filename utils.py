import collections
import sys
import ast
import pandas as pd
import torch
import os
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.optim import AdamW, SGD
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel, DistilBertTokenizer, DistilBertModel
from transformers import LlamaForCausalLM, LlamaTokenizerFast, LlamaModel
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from sklearn.utils import class_weight
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import _pickle as pickle
import time
from datetime import datetime, timedelta
import gc
import re
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel
import random
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def fitler_tl(labels, tweets):
    f = []
    for l, t in zip(labels, tweets):
        if l in ['baixo']:
            f.append(t)
    return f


def fitler_1to10(labels, tweets):
    f = []
    for l, t in zip(labels, tweets):
        try:
            if int(l) > 5:
                f.append(t)
        except:
            pass
    return f


def load_data(portion, task, sel):
    # df_treino_relevancia_1to10 = pickle.load(open(f'/home/wesley/Documents/projects/llm_experiments/sbr/train_{task}_relevancia_1to10_all.pkl', 'rb'))
    # df_teste_relevancia_1to10 = pickle.load(open(f'/home/wesley/Documents/projects/llm_experiments/sbr/teste_{task}_relevancia_1to10.pkl', 'rb'))
    # df_treino_relevancia = pickle.load(open(f'/home/wesley/Documents/projects/llm_experiments/sbr/treino_{task}_relevancia.pkl', 'rb'))
    # df_teste_relevancia = pickle.load(open(f'/home/wesley/Documents/projects/llm_experiments/sbr/teste_{task}_relevancia.pkl', 'rb'))
    PATH = '/home/wesley/Documents/doutorado/models/SetembroBR_HAN/data'
    df_c = pd.read_csv(f'{PATH}/{task}/{portion}_{task}_c_SetembroBR_v6.csv', sep=';')
    df_d = pd.read_csv(f'{PATH}/{task}/{portion}_{task}_SetembroBR_v6.csv', sep=';')
    df_c_net = pd.read_csv(f'{PATH}/{task}/{task}_ctrl-users-network_final_v6.csv', sep=';')
    df_d_net = pd.read_csv(f'{PATH}/{task}/{task}_diag-users-network_final_v6.csv', sep=';')
    df_d['Text'] = df_d['Text'].apply(lambda x: x.split('#'))
    df_c['Text'] = df_c['Text'].apply(lambda x: x.split('#'))
    # if sel not in ['all', 'all_br', 'balanced', 'medical_terms', 'pref_medical_terms', 'r80', 'dev80', 'r100']:
    #     df_d = df_d[df_d['User_ID'].isin(df_d_net[df_d_net['Event_Type'] == sel]['User_ID'])]
    #     df_c_net = df_c_net[df_c_net['Counterpart'].isin(df_d['User_ID'])]
    #     df_c = df_c[df_c['User_ID'].isin(df_c_net['User_ID'])]
    if sel == 'balanced' and portion == 'train':
        # return df_c, df_d, df_c_net
        dt = {'D':'depression', 'A': 'anxiety'}
        user_id_c_all = []
        for _, user_id in df_d.iterrows():
            user_id_c = df_c_net[df_c_net['Counterpart'] == user_id['User_ID']].sample(n=1).iloc[0]['User_ID']
            user_id_c_all.append(user_id_c)
        df_c = df_c[df_c['User_ID'].isin(set(user_id_c_all))]
    if sel == 'relevancia':
        if portion == 'test':
            d = dict(zip(df_teste_relevancia['User_ID'], df_teste_relevancia['label']))
            df_c['Text'] = df_c.apply(lambda x: fitler_tl(d[x['User_ID']], x['Text']), axis=1)
            df_d['Text'] = df_d.apply(lambda x: fitler_tl(d[x['User_ID']], x['Text']), axis=1)
        else:
            d = dict(zip(df_treino_relevancia['User_ID'], df_treino_relevancia['tl_prediction']))
            df_c['Text'] = df_c.apply(lambda x: fitler_tl(d[x['User_ID']], x['Text']), axis=1)
            df_d['Text'] = df_d.apply(lambda x: fitler_tl(d[x['User_ID']], x['Text']), axis=1)

    if sel == '1to10':
        if portion == 'test':
            d = dict(zip(df_teste_relevancia_1to10['User_ID'], df_teste_relevancia_1to10['label']))
            df_c['Text'] = df_c.apply(lambda x: fitler_1to10(d[x['User_ID']], x['Text']), axis=1)
            df_d['Text'] = df_d.apply(lambda x: fitler_1to10(d[x['User_ID']], x['Text']), axis=1)
        else:
            d = dict(zip(df_treino_relevancia_1to10['User_ID'], df_treino_relevancia_1to10['label']))
            df_c['Text'] = df_c.apply(lambda x: fitler_1to10(d[x['User_ID']], x['Text']), axis=1)
            df_d['Text'] = df_d.apply(lambda x: fitler_1to10(d[x['User_ID']], x['Text']), axis=1)

    return pd.concat((df_c, df_d))


def load_raw_data(task):
    PATH = '/home/wesley/Documents/doutorado/models/SetembroBR_HAN/data'
    df_c_train = pd.read_csv(f'{PATH}/{task}/train_{task}_c_SetembroBR_v6.csv', sep=';')
    df_d_train = pd.read_csv(f'{PATH}/{task}/train_{task}_SetembroBR_v6.csv', sep=';')
    df_d_train['Text'] = df_d_train['Text'].apply(lambda x: x.split('#'))
    df_c_train['Text'] = df_c_train['Text'].apply(lambda x: x.split('#'))
    df_train = pd.concat((df_c_train, df_d_train))
    df_c_test = pd.read_csv(f'{PATH}/{task}/test_{task}_c_SetembroBR_v6.csv', sep=';')
    df_d_test = pd.read_csv(f'{PATH}/{task}/test_{task}_SetembroBR_v6.csv', sep=';')
    df_d_test['Text'] = df_d_test['Text'].apply(lambda x: x.split('#'))
    df_c_test['Text'] = df_c_test['Text'].apply(lambda x: x.split('#'))
    df_test = pd.concat((df_c_test, df_d_test))
    return df_train, df_test


def load_timeseries_data(portion, task, sel):
    df_treino_relevancia_1to10 = pickle.load(open(f'/home/wesley/Documents/projects/llm_experiments/sbr/train_{task}_relevancia_1to10_all.pkl', 'rb'))
    df_teste_relevancia_1to10 = pickle.load(open(f'/home/wesley/Documents/projects/llm_experiments/sbr/teste_{task}_relevancia_1to10.pkl', 'rb'))
    df_treino_relevancia = pickle.load(open(f'/home/wesley/Documents/projects/llm_experiments/sbr/treino_{task}_relevancia.pkl', 'rb'))
    df_teste_relevancia = pickle.load(open(f'/home/wesley/Documents/projects/llm_experiments/sbr/teste_{task}_relevancia.pkl', 'rb'))
    PATH = '/home/wesley/Documents/doutorado/models/SetembroBR_HAN/data'
    df_c = pd.read_csv(f'{PATH}/{task}/{portion}_{task}_c_SetembroBR_v6.csv', sep=';')
    df_d = pd.read_csv(f'{PATH}/{task}/{portion}_{task}_SetembroBR_v6.csv', sep=';')
    df_c_net = pd.read_csv(f'{PATH}/{task}/{task}_ctrl-users-network_final_v6.csv', sep=';')
    df_d_net = pd.read_csv(f'{PATH}/{task}/{task}_diag-users-network_final_v6.csv', sep=';')
    df_d['Text'] = df_d['Text'].apply(lambda x: x.split('#'))
    df_c['Text'] = df_c['Text'].apply(lambda x: x.split('#'))
    dict_net_c = dict(zip(df_c_net['User_ID'].tolist(), df_c_net['Timeline'].apply(lambda x: eval(x)).tolist()))
    dict_net_d = dict(zip(df_d_net['User_ID'].tolist(), df_d_net['Timeline'].apply(lambda x: eval(x)).tolist()))

    df_c['timeline'] = df_c['User_ID'].apply(lambda x: dict_net_c[x])
    df_d['timeline'] = df_d['User_ID'].apply(lambda x: dict_net_d[x])

    if portion == 'test':
        d = dict(zip(df_teste_relevancia_1to10['User_ID'], df_teste_relevancia_1to10['label']))
        df_c['scores'] = df_c['User_ID'].apply(lambda x: d[x])
        df_d['scores'] = df_d['User_ID'].apply(lambda x: d[x])
    else:
        d = dict(zip(df_treino_relevancia_1to10['User_ID'], df_treino_relevancia_1to10['label']))
        df_c['scores'] = df_c['User_ID'].apply(lambda x: d[x])
        df_d['scores'] = df_d['User_ID'].apply(lambda x: d[x])

    return pd.concat((df_c, df_d))


# ao copiar/colar verifique se os caracteres de acentuaçao nas regexs estão corretos
def makeTags(text):
    text = str(text).lower()
    if re.search(r"diagn[oó]stic", text, flags=re.DOTALL):
        return 'diagnostico'
    if re.search(r"tratamento", text, flags=re.DOTALL):
        return 'tratamento'
    if re.search(r"depress[aã]o", text, flags=re.DOTALL):
        return 'depressao'
    if re.search(r"ansiedade", text, flags=re.DOTALL):
            return 'ansiedade'
    if re.search(r"depressivo|tarja\sp|ansiol[íi]tico", text, flags=re.DOTALL):
        return 'antidepress'
    if re.search(r"m[eé]dic[oa]|psic[oó]l[oó]g[oa]|psiquiatra|neurologista|terapeuta", text, flags=re.DOTALL):
        return 'medico'
    return ''


# exemplo de uso - marca todos os textos que contêm palavras de natureza clínica
def filter_medical_terms(texts):
    sel_texts_ = []
    for text_list in tqdm(texts):
        sel_texts = [tt for tt in text_list if makeTags(tt) == '']
        sel_texts_.append(sel_texts)
    return sel_texts_


# exemplo de uso - marca todos os textos que contêm palavras de natureza clínica
def sel_medical_terms(texts):
    size = 100
    sel_texts_ = []
    for text_list in tqdm(texts):
        sel_texts = [tt for tt in text_list if makeTags(tt) == '']
        med_sel_texts = [tt for tt in text_list if makeTags(tt) != '']
        if size <= len(med_sel_texts):
            sel_texts_.append(med_sel_texts[-size:])
        else:
            l = size - len(med_sel_texts)
            # print(l)
            if len(sel_texts) > l:
                add_texts = random.sample(sel_texts, l)
                med_sel_texts = med_sel_texts + add_texts
                random.shuffle(med_sel_texts)
                sel_texts_.append(med_sel_texts)
            else:
                sel_texts_.append(med_sel_texts + sel_texts)

    return sel_texts_


def random_cont_selection(texts, n=80):
    sel_texts_ = []
    for text_list in tqdm(texts):
        if len(text_list) > n + 1:
            fp = random.randint(n, len(text_list))
            sp = fp - n
        else:
            sp = 0
            fp = len(text_list)
        sel_texts_.append(text_list[sp:fp])

    return sel_texts_


def recent_selection(texts, n=80):
    sel_texts_ = []
    for text_list in tqdm(texts):
        sel_texts_.append(text_list[-n:])

    return sel_texts_


def load_convert(task, sel):
    df_test = load_data('test', task, sel)
    df_train = load_data('train', task, sel)
    # df_test['Text'] = df_test['Text'].apply(lambda x: x.split('#'))
    df_test['label'] = df_test['Diagnosed_YN'].apply(lambda x: 1 if x == 'yes' else 0)
    # df_train['Text'] = df_train['Text'].apply(lambda x: x.split('#'))
    df_train['label'] = df_train['Diagnosed_YN'].apply(lambda x: 1 if x == 'yes' else 0)
    train_initial_size = df_train['Text'].apply(lambda x: len(x)).sum()
    test_initial_size = df_test['Text'].apply(lambda x: len(x)).sum()
    random.seed(42)
    if sel == 'medical_terms':
        print('Start filter')
        df_train['Text'] = filter_medical_terms(df_train['Text'])
        df_test['Text'] = filter_medical_terms(df_test['Text'])
        train_final_size = df_train['Text'].apply(lambda x: len(x)).sum()
        test_final_size = df_test['Text'].apply(lambda x: len(x)).sum()
        print(f'On task {task}.\n\tThe number of samples in the train dataset is ')
    elif sel == 'pref_medical_terms':
        print('sel', sel)
        df_train['Text'] = sel_medical_terms(df_train['Text'])
        df_test['Text'] = sel_medical_terms(df_test['Text'])
        train_final_size = df_train['Text'].apply(lambda x: len(x)).sum()
        test_final_size = df_test['Text'].apply(lambda x: len(x)).sum()
        print(f'On task {task}.\n\tThe number of samples in the train dataset is ', len(df_train))
    elif sel == 'r80':
        print('sel', sel)
        df_train['Text'] = random_cont_selection(df_train['Text'])
        # df_test['Text'] = random_cont_selection(df_test['Text'])
    elif sel == 'r100':
        print('sel', sel)
        df_train['Text'] = random_cont_selection(df_train['Text'], 100)
        # df_test['Text'] = random_cont_selection(df_test['Text'])
    elif sel == 'dev80':
        print('sel', sel)
        df_train['Text'] = recent_selection(df_train['Text'])
        # df_test['Text'] = random_cont_selection(df_test['Text'])
    return df_train, df_test


def load_convert_timeseries(task, sel):
    df_test = load_timeseries_data('test', task, sel)
    df_train = load_timeseries_data('train', task, sel)
    # df_test['Text'] = df_test['Text'].apply(lambda x: x.split('#'))
    df_test['label'] = df_test['Diagnosed_YN'].apply(lambda x: 1 if x == 'yes' else 0)
    # df_train['Text'] = df_train['Text'].apply(lambda x: x.split('#'))
    df_train['label'] = df_train['Diagnosed_YN'].apply(lambda x: 1 if x == 'yes' else 0)
    return df_train, df_test


class CustomTextDataset(Dataset):
    def __init__(self, tokens_mask, labels):
        self.tokens_mask = tokens_mask
        self.labels = labels
        self.size = 10
        random.seed(42)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.tokens_mask[idx]
        start = random.randint(0, data['tokens'].shape[0] - self.size)
        data_short = {v: data[v][start: start + self.size] for v in data}
        return [data_short, label]


class CustomTimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CustomTextDatasetCombiner(Dataset):
    def __init__(self, tokens_mask, labels, xtrain):
        self.tokens_mask = tokens_mask
        self.labels = labels
        self.size = 10
        self.ps = xtrain
        random.seed(42)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.tokens_mask[idx]
        ps = self.ps[idx]
        return [data, ps, label]


class CustomTextDatasetCombinerBert(Dataset):
    def __init__(self, tokens, masks, labels, xtrain):
        self.tokens = tokens
        self.masks = masks
        self.labels = labels
        self.ps = xtrain
        random.seed(42)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        try:
            t = self.tokens[idx]
        except:
            print(1)
        m = self.masks[idx]
        ps = self.ps[idx]
        return [t, m, ps, label]


class CustomTextDatasetALL(Dataset):
    def __init__(self, tokens_mask):
        self.tokens_mask = tokens_mask
        random.seed(42)

    def __len__(self):
        return len(self.tokens_mask)

    def __getitem__(self, idx):
        data = self.tokens_mask[idx]
        # data_short = {v: data[v][start: start + self.size] for v in data}
        return data


class CustomTextDatasetIndividual(Dataset):
    def __init__(self, tokens_mask, label):
        self.tokens_mask = tokens_mask
        self.label = label
        self.size = 10
        self.start = 0

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        label = self.label
        data_short = {v: self.tokens_mask[v][self.start: self.start + self.size] for v in self.tokens_mask}
        self.start+= self.size
        return [data_short, label, self.start > len(self.tokens_mask['user_posts'])]


class CustomTextDatasetSeq(Dataset):
    def __init__(self, tokens_mask, labels):
        self.tokens_mask = tokens_mask
        self.labels = labels
        self.size = 10
        self.start = {v: 0 for v in range(len(self.labels))}
        random.seed(42)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.tokens_mask[idx]
        data_short = {v: data[v][self.start[idx]: min(self.start[idx] + self.size, len(data[v]))] for v in data}
        self.start[idx] += self.size
        # print(self.start[idx], max([len(data[v]) for v in data]))
        if self.start[idx] >= max([len(data[v]) for v in data]):
            self.start[idx] = 0
        return [data_short, label]


class CustomTextDatasetText(Dataset):
    def __init__(self, tokens_mask, texts, labels):
        self.tokens_mask = tokens_mask
        self.labels = labels
        self.texts = texts
        self.size = 10
        random.seed(42)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.tokens_mask[idx]
        start = random.randint(0, data['tokens'].shape[0] - self.size)
        data_short = {v: data[v][start: start + self.size] for v in data}
        print(type(self.texts[idx]), self.texts[idx])
        # print(type(self.texts[idx]), self.texts[idx])
        texts = self.texts[idx][start: start + self.size]
        # print(texts)
        return [data_short, texts, label]



def get_max_possible_size(x):
    return 10*int(x/10)


def create_weight_dict(labels):
    unique_labels = np.unique(labels)
    class_weights = class_weight.compute_class_weight('balanced', classes=unique_labels, y=labels)
    return class_weights


def get_tokens_mask_dates_split(X=None, y=None, tokenizer=None, type_data=None, task=None, sel=None, ml=30):
    path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/data/{type_data}_{sel}_{task}_processed.pkl'

    if os.path.isfile(path):
        print('load', path)
        # 0/0
        X_processed, y = pickle.load(open(path, 'rb'))
    else:
        print('Prepare')
        X_processed = []
        for user_posts in tqdm(X):
            # print(tokenizer)
            user_posts = user_posts[-get_max_possible_size(len(user_posts)):]
            if len(user_posts) == 0:
                X_processed.append(
                    {'tokens': torch.tensor([]),
                     'mask': torch.tensor([]),
                     }
                )
            else:
                tokenized_texts = tokenizer.batch_encode_plus(user_posts, max_length=ml, padding='max_length', truncation=True)
                tokens_tensor = torch.tensor(tokenized_texts['input_ids'])
                segments_tensors = torch.tensor(tokenized_texts['attention_mask'])
                X_processed.append(
                    {'tokens': tokens_tensor,
                     'mask': segments_tensors,
                     }
                )
        pickle.dump([X_processed, y], open(path, 'wb'))
    return X_processed, y


def get_dataloader_text(X_, X_raw, y_, shuffle):
    batch_size = 16
    TD_ = CustomTextDatasetText(X_, X_raw, y_)
    return DataLoader(TD_, batch_size=batch_size, shuffle=shuffle, collate_fn=(lambda x: x))


def get_dataloader(X_, y_, shuffle):
    batch_size = 16
    TD_ = CustomTextDataset(X_, y_)
    return DataLoader(TD_, batch_size=batch_size, shuffle=shuffle, collate_fn=(lambda x: x))


def get_dataloader_combiner(X_, y_, xtrain, shuffle):
    batch_size = 16
    TD_ = CustomTextDatasetCombiner(X_, y_, xtrain)
    return DataLoader(TD_, batch_size=batch_size, shuffle=shuffle, collate_fn=(lambda x: x))


def get_dataloader_individual(X_, y_, shuffle):
    batch_size = 32
    TD_ = CustomTextDatasetIndividual(X_, y_)
    return DataLoader(TD_, batch_size=batch_size, shuffle=shuffle, collate_fn=(lambda x: x))


def get_dataloader_all(X_, shuffle):
    batch_size = 16
    TD_ = CustomTextDatasetALL(X_)
    return DataLoader(TD_, batch_size=batch_size, shuffle=shuffle, collate_fn=(lambda x: x))



def load_checkpoint(load_path, model):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location='cuda')
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def split_tensor(tensor, chunk_size):
    n = tensor.size(0)
    chunks = []
    for i in range(0, n, chunk_size):
        if i + chunk_size <= n:
            chunks.append(tensor[i:i + chunk_size])
        else:
            chunks.append(tensor[i:])
    return chunks


def main_mcnemar(c1, c2):
    df = pd.concat([c1,c2],axis=1) # lado
    df.columns = ['c1','c2']

    df['right_right'] = df.apply(lambda x: int(x.c1==1 and x.c2==1),axis=1)
    df['right_wrong'] = df.apply(lambda x: int(x.c1==1 and x.c2==0),axis=1)
    df['wrong_right'] = df.apply(lambda x: int(x.c1==0 and x.c2==1),axis=1)
    df['wrong_wrong'] = df.apply(lambda x: int(x.c1==0 and x.c2==0),axis=1)

    cell_11 = df.right_right.sum()
    cell_12 = df.right_wrong.sum()
    cell_21 = df.wrong_right.sum()
    cell_22 = df.wrong_wrong.sum()

    table = [
             [cell_11, cell_12],
             [cell_21, cell_22]
            ]

    if cell_11<25 or cell_12<25 or cell_21<25 or cell_22<25:
        result = mcnemar(table, exact=True)
    else:
        result = mcnemar(table, exact=False, correction=True)

    # print('\n'+corpus+"_"+classifier1+"_"+classifier2+ ' : ',end='')
    print('stat=%.3f, p=%.7f' % (result.statistic, result.pvalue),end='')

    significant = False
    for alpha in [0.001, 0.01, 0.05]:
        if result.pvalue < alpha:
            print(' p<' + str(alpha))
            significant = True
            break
    if not significant:
        print(' not significant')
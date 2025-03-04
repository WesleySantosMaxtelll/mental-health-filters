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
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel
from utils import *
from models import *
from sklearn.model_selection import train_test_split

# df_train_D = pickle.load(open(f'converted_dataframes/D_train_converted.pkl', 'rb'))
# df_test_D = pickle.load(open(f'converted_dataframes/D_test_converted.pkl', 'rb'))
# df_train_A = pickle.load(open(f'converted_dataframes/A_train_converted.pkl', 'rb'))
# df_test_A = pickle.load(open(f'converted_dataframes/A_test_converted.pkl', 'rb'))
# dfs = [df_test_D, df_train_D, df_test_A, df_train_A]
# alto = 0
# medio = 0
# baixo = 0
# for df in dfs:
#     for tl in df['alto_medio_baixo'].tolist():
#         for label in tl:
#             if label == 'alto':
#                 alto += 1
#             elif label == 'medio':
#                 medio += 1
#             else:
#                 baixo += 1


torch.cuda.empty_cache()
device = torch.device('cuda')
print(device)
# from keras.preprocessing.sequence import pad_sequences
source_folder = 'Data'
destination_folder = 'Model'


# Define the function for creating a weight dictionary.


def get_loss(outputs, b_labels, vect_weights):
    loss_fct = CrossEntropyLoss(weight=torch.tensor(vect_weights, dtype=torch.float32))
    loss_fct.to(device)
    loss = loss_fct(outputs[1].view(-1, 2), b_labels.view(-1))

    return loss


class Trainer():
    def __init__(self):
        pass

    def get_relevancia_baixo(self, x):
        texts = []
        for t, l in zip(x['Text'], x['alto_medio_baixo']):
            if l == 'baixo':
                texts.append(t)
        return texts


    def get_relevancia_alto_medio(self, x):
        texts = []
        for t, l in zip(x['Text'], x['alto_medio_baixo']):
            if l == 'alto' or l == 'medio':
                texts.append(t)
        return texts


    def run_train(self, task, sel, model_name, type_):
        print(task, sel, model_name)
        try:
            print('Load')
            # 1/0
            X_test, y_test = get_tokens_mask_dates_split(None, None, None, f'{type_}_test', task, sel)
            X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'{type_}_train', task, sel)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
        except:
            print('Create')
            df_train = pickle.load(open(f'converted_dataframes/{task}_train_converted.pkl', 'rb'))
            df_test = pickle.load(open(f'converted_dataframes/{task}_test_converted.pkl', 'rb'))
            df_train['label'] = (df_train['Diagnosed_YN'] == 'yes').astype(int)
            df_test['label'] = (df_test['Diagnosed_YN'] == 'yes').astype(int)
            if sel == 'relevancia_baixo':
                df_train['filter_text'] = df_train.apply(lambda x: self.get_relevancia_baixo(x), axis=1)
                df_test['filter_text'] = df_test.apply(lambda x: self.get_relevancia_baixo(x), axis=1)
            if sel == 'relevancia_alto_medio':
                df_train['filter_text'] = df_train.apply(lambda x: self.get_relevancia_alto_medio(x), axis=1)
                df_test['filter_text'] = df_test.apply(lambda x: self.get_relevancia_alto_medio(x), axis=1)
            print(len(df_train), len(df_test))
            # return df_train, df_test
            # tokenizer = BertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased')
            # tokenizer = BertTokenizerFast.from_pretrained('pablocosta/bertweet-br-base-uncased')
            # tokenizer = LlamaTokenizerFast.from_pretrained("pablocosta/llama-7b", use_auth_token="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", load_in_8bit=True)
            tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertweet-br-base-uncased")
            X_train, y_train = get_tokens_mask_dates_split(df_train['filter_text'].tolist(), df_train['label'].tolist(),
                                                           tokenizer, f'{type_}_train', task, sel)
            X_test, y_test = get_tokens_mask_dates_split(df_test['filter_text'].tolist(), df_test['label'].tolist(), tokenizer,
                                                         f'{type_}_test', task, sel)
            # return X_train, y_train
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
        # return X_train, y_train
        y_train = [el for el, x in zip(y_train, X_train) if len(x['tokens']) > 10]
        X_train = [el for el in X_train if len(el['tokens']) > 10]
        y_val = [el for el, x in zip(y_val, X_val) if len(x['tokens']) > 10]
        X_val = [el for el in X_val if len(el['tokens']) > 10]
        dataloader_train = get_dataloader(X_train, y_train, False)
        dataloader_val = get_dataloader(X_val, y_val, False)
        dataloader_test = get_dataloader(X_test, y_test, False)
        # return
        device = 'cuda'
        if model_name == 'lstm':
            model = BertUser(device)
        if model_name == 'bert_twitter':
            model = BertUserTwitter(device)
        elif model_name == 'moe':
            model = MoE(device)
        elif model_name == 'moe_br':
            model = MoEBR(device)
        elif model_name == 'ens':
            model = Ensemble(device)
        elif model_name == 'ens-concat':
            model = EnsembleConcat(device)
        elif model_name == 'bart_twitter':
            model = BartUserTwitter(device)
        else:
            raise Exception('Model not defined')

        model.to(device)
        epoch_wait = 0
        # file_path = f'/content/drive/MyDrive/doutorado/diag_trat_saving/{sel}/ALL_DATA-model-{model_name}-{task}-sel-{sel}-epoch{epoch_wait}.pt'
        # print('*' * 20)
        # load_checkpoint(file_path, model)

        lr = 1e-5
        optimizer = AdamW(model.parameters(), lr=lr)

        train_loss_set = []
        epochs = 50
        file_path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/'
        vect_weights = create_weight_dict(y_train)
        print(vect_weights)
        loss_fct = CrossEntropyLoss(weight=torch.tensor(vect_weights, dtype=torch.float32), reduction='mean')
        loss_fct.to(device)
        for epoch in range(epochs):
            # if epoch_wait > epoch:
            #     continue
            print(f'Epoch {epoch}')
            model.train()
            pred, true_label = [], []
            print(len(dataloader_train))
            for i, batch in enumerate(dataloader_train):
                if i % 30 == 0:
                    # break
                    print(datetime.fromtimestamp(time.time()), i, train_loss_set[-1] if len(train_loss_set) else 0)
                # continue
                atts = torch.stack([aa[0]['tokens'] for aa in batch]).to(device)
                masks = torch.stack([aa[0]['mask'] for aa in batch]).to(device)
                outputs = model(atts, masks)
                loss = loss_fct(outputs, torch.tensor([b[1] for b in batch]).to(device))
                # output = loss(torch.argmax(outputs[1], dim=1).add_(-1), b_labels.add_(-1))
                train_loss_set.append(loss.item())
                # Backward pass
                loss.backward()
                # Update parameters and take a step using the computed gradient
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()

                pred = pred + outputs.cpu().argmax(1).tolist()
                true_label = true_label + [b[1] for b in batch]
                gc.collect()
                torch.cuda.empty_cache()
            # continue
            print('Results')
            print(classification_report(true_label, pred))
            # continue
            model.eval()

            with torch.no_grad():
                # validation loop
                pred, real = [], []
                for j, batch in enumerate(dataloader_val):
                    atts = torch.stack([aa[0]['tokens'] for aa in batch]).to(device)
                    masks = torch.stack([aa[0]['mask'] for aa in batch]).to(device)
                    if model_name == 'ens':
                        outputs, model_outputs = model(atts, masks)
                    else:
                        outputs = model(atts, masks)

                    pred = pred + outputs.cpu().argmax(1).tolist()
                    real = real + [b[1] for b in batch]
                print(classification_report(real, pred))
                save_checkpoint(file_path + '/' + f'ALL_DATA-model-{model_name}-{task}-sel-{sel}-baixo-epoch{epoch}_v2.pt',
                                model, 0)
                # save_metrics(file_path + '/' + f'metrics-full-{task}-epoch{epoch}.pt', 0,0,0)
            model.train()


Trainer().run_train('A', 'relevancia_baixo', 'bert_twitter', 'Bert_relevancia_baixo')
# Trainer().run_train('A', 'relevancia_alto_medio', 'bert_twitter', 'Bert_relevancia_alto_medio')

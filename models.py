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
from transformers import BertModel
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
import random

class BertUser(nn.Module):
    def __init__(self, device, output_size=2):
        super(BertUser, self).__init__()
        self.device = device
        pre_trained = 'neuralmind/bert-base-portuguese-cased'
        self.bert = BertModel.from_pretrained(pre_trained)
        self.hidden_size = self.bert.config.hidden_size
        self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=100,
                           num_layers=4, bidirectional=True)
        self.sample_size = 10
        self.fc_out3 = nn.Linear(6000, 1000)
        self.fc_out2 = nn.Linear(1000, 100)
        self.fc_out1 = nn.Linear(100 * self.sample_size, output_size)

    def forward(self, tokens, masks):
        outputs = []
        # iterate in the batch through all sequences
        for token, mask in zip(tokens, masks):
            encoded = self.bert(input_ids=token, attention_mask=mask)['last_hidden_state']
            out, hiden = self.rnn(encoded)
            out = self.fc_out3(out.view(self.sample_size, -1))
            out = F.dropout(out, 0.1)
            out = self.fc_out2(out)
            out = F.dropout(out, 0.1)
            out = self.fc_out1(out.view(-1))
            out = torch.softmax(out, -1)
            outputs.append(out.squeeze())
        outputs = torch.stack(outputs)
        return outputs


class BertUserTwitter(nn.Module):
    def __init__(self, device, output_size=2):
        super(BertUserTwitter, self).__init__()
        self.device = device
        # pre_trained = 'neuralmind/bert-base-portuguese-cased'
        pre_trained = 'pablocosta/bertweet-br-base-uncased'
        self.bert = BertModel.from_pretrained(pre_trained)
        # self.bert = LlamaModel.from_pretrained("pablocosta/llama-7b", use_auth_token="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", load_in_8bit=True, device_map='auto')
        self.hidden_size = self.bert.config.hidden_size
        self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=100,
                           num_layers=4, bidirectional=True)
        self.sample_size = 10
        self.fc_out3 = nn.Linear(6000, 1000)
        self.fc_out2 = nn.Linear(1000, 100)
        self.fc_out1 = nn.Linear(100 * self.sample_size, output_size)

    def forward(self, tokens, masks):
        outputs = []
        # iterate in the batch through all sequences
        for token, mask in zip(tokens, masks):
            encoded = self.bert(input_ids=token, attention_mask=mask)['last_hidden_state']
            out, hiden = self.rnn(encoded)
            out = self.fc_out3(out.view(self.sample_size, -1))

            out = F.dropout(out, 0.1)
            out = self.fc_out2(out)
            out = F.dropout(out, 0.1)
            out = self.fc_out1(out.view(-1))
            out = torch.softmax(out, -1)
            outputs.append(out.squeeze())
        outputs = torch.stack(outputs)
        return outputs


class BertUserTwitterAM(nn.Module):
    def __init__(self, device, output_size=2):
        super(BertUserTwitterAM, self).__init__()
        self.device = device
        # pre_trained = 'neuralmind/bert-base-portuguese-cased'
        pre_trained = "pablocosta/bertabaporu-base-uncased"
        self.bert = BertModel.from_pretrained(pre_trained)
        # self.bert = LlamaModel.from_pretrained("pablocosta/llama-7b", use_auth_token="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", load_in_8bit=True, device_map='auto')
        self.hidden_size = self.bert.config.hidden_size
        self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=100,
                           num_layers=4, bidirectional=True)

        self.fc_out3 = nn.Linear(24000, 1000)
        self.fc_out2 = nn.Linear(1000, 100)
        self.fc_out1 = nn.Linear(100, output_size)

    def forward(self, tokens, masks):
        outputs = []
        # iterate in the batch through all sequences
        for token, mask in zip(tokens, masks):
            encoded = self.bert(input_ids=token, attention_mask=mask)['last_hidden_state']
            out, hiden = self.rnn(encoded)
            out = self.fc_out3(hiden[0].flatten())
            out = F.dropout(out, 0.1)
            out = self.fc_out2(out)
            out = F.dropout(out, 0.1)
            out = self.fc_out1(out.view(-1))
            out = torch.softmax(out, -1)
            outputs.append(out.squeeze())
        outputs = torch.stack(outputs)
        return outputs


class ExpertGate(nn.Module):
    def __init__(self, device, s, pre_trained='Geotrend/distilbert-base-pt-cased', output_size=2):
        super(ExpertGate, self).__init__()
        self.device = device
        self.hidden_size = s
        self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=100,
                           num_layers=4, bidirectional=True)
        self.sample_size = 10
        self.fc_out3 = nn.Linear(6000, 1000)
        self.fc_out2 = nn.Linear(1000, 100)
        self.fc_out1 = nn.Linear(100 * self.sample_size, output_size)

    def forward(self, bert_vectors):
        outputs = []
        # iterate in the batch through all sequences
        for encoded in bert_vectors:
            # encoded = self.bert(input_ids=token, attention_mask=mask)['last_hidden_state']
            out, hiden = self.rnn(encoded)
            out = self.fc_out3(out.view(self.sample_size, -1))
            out = F.dropout(out, 0.5)
            out = self.fc_out2(out)
            out = F.dropout(out, 0.5)
            out = self.fc_out1(out.view(-1))
            out = torch.softmax(out, -1)
            outputs.append(out.squeeze())
        outputs = torch.stack(outputs)
        return outputs


class MoE(nn.Module):
    def __init__(self, device, exps=3):
        super(MoE, self).__init__()
        pre_trained = 'neuralmind/bert-base-portuguese-cased'
        self.bert = BertModel.from_pretrained(pre_trained)
        self.exps = exps
        self.device = device
        self.nets = nn.ModuleList([
          ExpertGate(device, self.bert.config.hidden_size) for _ in range(self.exps)
        ])
        self.gating = ExpertGate(device, self.bert.config.hidden_size, output_size=exps)

    def forward(self, atts, masks):
        bert_out = []
        for token, mask in zip(atts, masks):
            # print(1)
            bert_vectors = self.bert(input_ids=token, attention_mask=mask)['last_hidden_state']
            bert_out.append(bert_vectors)
        # print(2)
        weights_sof = self.gating(bert_out)
        # print(3)
        preds = []
        for n in self.nets:
            # print(4)
            preds.append(n(bert_out))
        # print(5)
        return (torch.stack(preds) * weights_sof.T[..., None]).sum(0)


class MoEBR(nn.Module):
    def __init__(self, device, exps=3):
        super(MoEBR, self).__init__()
        pre_trained = 'pablocosta/bertweet-br-base-uncased'
        self.bert = BertModel.from_pretrained(pre_trained)
        self.exps = exps
        self.device = device
        self.nets = nn.ModuleList([
          ExpertGate(device, self.bert.config.hidden_size) for _ in range(self.exps)
        ])
        self.gating = ExpertGate(device, self.bert.config.hidden_size, output_size=exps)

    def forward(self, atts, masks):
        bert_out = []
        for token, mask in zip(atts, masks):
            # print(1)
            bert_vectors = self.bert(input_ids=token, attention_mask=mask)['last_hidden_state']
            bert_out.append(bert_vectors)
        # print(2)
        weights_sof = self.gating(bert_out)
        # print(3)
        preds = []
        for n in self.nets:
            # print(4)
            preds.append(n(bert_out))
        # print(5)
        return (torch.stack(preds) * weights_sof.T[..., None]).sum(0)


class Ensemble(nn.Module):
    def __init__(self, device):
        super().__init__()
        pre_trained = 'neuralmind/bert-base-portuguese-cased'
        self.bert = BertModel.from_pretrained(pre_trained)
        self.nets = nn.ModuleList()
        for _ in range(3):
            self.nets.append(ExpertGate(device, self.bert.config.hidden_size))
        # self.gate = MixedBertModel(4)

    def forward(self, atts, masks):
        bert_out = []
        for token, mask in zip(atts, masks):
            # print(1)
            bert_vectors = self.bert(input_ids=token, attention_mask=mask)['last_hidden_state']
            bert_out.append(bert_vectors)

        preds = []
        for n in self.nets:
            # print(4)
            preds.append(n(bert_out))

        with torch.no_grad():

            ens_outs = torch.stack(preds).mean(0)
        # gate_output = self.gate(input_ids=inputs[0], attention_mask=inputs[1])

        # ensemble output and the models output
        return ens_outs, preds


class EnsembleConcat(nn.Module):
    def __init__(self, device):
        super().__init__()
        pre_trained = 'neuralmind/bert-base-portuguese-cased'
        self.bert = BertModel.from_pretrained(pre_trained)
        self.nets = nn.ModuleList()
        for _ in range(3):
            self.nets.append(ExpertGate(device, self.bert.config.hidden_size))
        # self.gate = MixedBertModel(4)

    def forward(self, atts, masks):
        bert_out = []
        for token, mask in zip(atts, masks):
            # print(1)
            bert_vectors = self.bert(input_ids=token, attention_mask=mask)['last_hidden_state']
            bert_out.append(bert_vectors)

        preds = []
        for n in self.nets:
            # print(4)
            preds.append(n(bert_out))

        # with torch.no_grad():

        ens_outs = torch.stack(preds).mean(0)
        # gate_output = self.gate(input_ids=inputs[0], attention_mask=inputs[1])

        # ensemble output and the models output
        return ens_outs


class Combiner(nn.Module):
    def __init__(self, device, output_size=2):
        super(Combiner, self).__init__()
        self.device = device
        pre_trained = 'pablocosta/bertweet-br-base-uncased'
        self.bert = BertModel.from_pretrained(pre_trained)
        self.hidden_size = self.bert.config.hidden_size
        self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=100,
                           num_layers=4, bidirectional=True)
        self.sample_size = 10
        self.fc_out3 = nn.Linear(6000, 1000)
        self.fc_out2 = nn.Linear(1000, 100)
        self.fc_out1 = nn.Linear(100 * self.sample_size + 3, output_size)  # +2 for v1 and v2

        # Attention mechanism parameters
        self.attention_linear = nn.Linear(1003, 1)  # Linear layer for attention calculation
        self.softmax = nn.Softmax(dim=1)  # Softmax layer for attention weights

    def forward(self, tokens, masks, v1, v2, v3):
        outputs = []
        # iterate in the batch through all sequences
        for token, mask, v1_val, v2_val, v3_val in zip(tokens, masks, v1, v2, v3):
            encoded = self.bert(input_ids=token, attention_mask=mask)['last_hidden_state']
            out, hidden = self.rnn(encoded)
            out = self.fc_out3(out.view(self.sample_size, -1))
            out = F.dropout(out, 0.1)
            out = self.fc_out2(out)
            out = F.dropout(out, 0.1)
            # Concatenate v1 and v2 with the output before passing to final layer
            combined_input = torch.cat((out.view(-1), torch.tensor([v1_val]).cuda().float(), torch.tensor([v2_val]).cuda().float(), torch.tensor([v3_val]).cuda().float()))


            # Attention mechanism
            # attn_weights = self.softmax(self.attention_linear(combined_input.unsqueeze(0).unsqueeze(0))).squeeze()
            # attn_output = torch.sum(out * attn_weights.unsqueeze(-1), dim=0)

            out = self.fc_out1(combined_input)
            out = torch.softmax(out, -1)
            outputs.append(out.squeeze())
        outputs = torch.stack(outputs)
        return outputs


class GatingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_models):
        super(GatingNetwork, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_models)

    def forward(self, x):
        x = self.embedding(x)
        # x = torch.mean(x, dim=1)  # Mean pooling over the sequence dimension
        x = F.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), 1)  # Sigmoid activation to get gating weights
        return x


class CombinedModel(nn.Module):
    def __init__(self, input_size, num_models):
        super(CombinedModel, self).__init__()
        self.num_models = num_models
        self.gating_network = GatingNetwork(input_size=input_size, hidden_size=128, num_models=num_models)

    def forward(self, x, predictions):
        gating_weights = self.gating_network(x)
        out = torch.mul(gating_weights, predictions).sum(1)
        return torch.stack([1 - out, out], dim=1).float()

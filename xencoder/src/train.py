import sys
from .translation import TranslationModel
from ..data.dataloader import ParallelDataset, PadSequence

import argparse
from time import time

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from transformers import XLMRobertaModel, XLMRobertaTokenizer

from sklearn.manifold import TSNE
import seaborn as sns

seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(model, train_loader):

    model = TranslationModel()
    model.eval()

    src_tensors = []
    tgt_tensors = []

    for batch_no, batch in enumerate(train_loader):
        source = batch[0]
        target = batch[1]

        model.zero_grad()        

        source, target = model(source, target)
        src_tensors.append(source)
        tgt_tensors.append(target)

    src_tensors = torch.cat(src_tensors, 0)
    tgt_tensors = torch.cat(tgt_tensors, 0) 
    
    pivot_adapter = torch.mm(torch.pinverse(src_tensors), tgt_tensors)
    src_transformed = torch.mm(src_tensors, pivot_adapter)
   

    """
    src_prior = TSNE(n_components=2).fit_transform(src_tensors.detach().numpy())
    src_embedded = TSNE(n_components=2).fit_transform(src_transformed.detach().numpy())
    pvt_embedded = TSNE(n_components=2).fit_transform(tgt_tensors.detach().numpy())


    sns.scatterplot(x=src_prior[:,0], y=src_prior[:, 1], markers='+')
    sns.scatterplot(x=src_embedded[:,0], y=src_embedded[:, 1], markers='*')
    sns.scatterplot(x=pvt_embedded[:,0], y=pvt_embedded[:, 1], markers='-') 
    """

    return pivot_adapter, src_transformed


def build_model():
    
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    model = TranslationModel()

    return model, tokenizer


def run_experiment(src_train_path, pvt_train_path):

    model, tokenizer = build_model()
    
    custom_pad_sequence = PadSequence(tokenizer.pad_token_id) 

    train_loader = DataLoader(ParallelDataset(tokenizer, src_train_path, pvt_train_path),
                    shuffle=True,
                    batch_size=64,
                    collate_fn=custom_pad_sequence)
    pivot_adapter, transformed_src = train(model, train_loader)
    
    return pivot_adapter, transformed_src


src_path = sys.argv[1]
tgt_path = sys.argv[2]
run_experiment(src_path, tgt_path)

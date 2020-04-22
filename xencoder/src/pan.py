import argparse
import pickle as pkl

from scipy import linalg
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaModel, XLMRobertaTokenizer

from ..data.dataloader import PadSequence, ParallelDataset

seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

def train(source, target):
    
    model = XLMRobertaModel.from_pretrained('xlm-roberta-large')

    pad_sequence = PadSequence(1)
    train_loader = DataLoader(ParallelDataset(source, target),
                              shuffle=True,
                              batch_size=64,
                              collate_fn=pad_sequence)
    
    #Set requires_grad to False.
    for name, params in model.named_parameters():
        params.requires_grad = False

    model.eval()

    src_embeddings = None
    tgt_embeddings = None

    for source, target in train_loader:

        model.zero_grad()        

        source = model(source)[0].mean(axis=1)
        target = model(target)[0].mean(axis=1)
       
        src_embeddings =  source.detach().numpy() if src_embeddings is None else np.concatenate((src_embeddings, source.detach().numpy()), axis=0)
        tgt_embeddings =  target.detach().numpy() if tgt_embeddings is None else np.concatenate((tgt_embeddings, target.detach().numpy()), axis=0)

    pivot_adapter = linalg.orthogonal_procrustes(src_embeddings, tgt_embeddings)[0]

    return pivot_adapter

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pre-Training the Pivot Adapter Network')

    parser.add_argument('--source', type=str, help='Path to source corpus')
    parser.add_argument('--target', type=str, help='Path to target corpus')
    parser.add_argument('--save_path', type=str, help='Destination to store the adapter')

    args = parser.parse_args()

    pivot_adapter = train(args.source, args.target)

    with open(args.save_path + args.source.split('.')[-2] + args.target.split('.')[-2] + '.pkl', 'wb') as handle:
        pkl.dump(pivot_adapter, handle)

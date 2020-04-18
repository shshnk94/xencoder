import argparse

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

def build_model():

    #XLM-RoBERTa from huggingface/transformer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    encoder = XLMRobertaModel.from_pretrained('xlm-roberta-large')

    return tokenizer, encoder

def train(source, target):
    
    tokenizer, model = build_model()

    pad_sequence = PadSequence(tokenizer.pad_token_id)
    train_loader = DataLoader(ParallelDataset(tokenizer, source, target),
                              shuffle=True,
                              batch_size=64,
                              collate_fn=pad_sequence)

    model.eval()
    src_embeddings = []
    tgt_embeddings = []

    for batch in train_loader:
        source = batch[0]
        target = batch[1]

        model.zero_grad()        

        source = model(source)
        target = model(target)

        src_embeddings.append(source)
        tgt_embeddings.append(target)

    
    src_embeddings = torch.cat(src_embeddings, 0)
    tgt_embeddings = torch.cat(tgt_embeddings, 0)

    pivot_adapter = linalg.orthogonal_procrustes((src_embeddings.detach().numpy()), (tgt_embeddings.detach().numpy()))[0]

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pre-Training the Pivot Adapter Network')

    parser.add_argument('--source', type=str, help='Path to source corpus')
    parser.add_argument('--target', type=str, help='Path to target corpus')
    parser.add_argument('--save_path', type=str, help='Destination to store the adapter')

    args = parser.parse_args()

    pivot_adapter = train(args.source, args.target)

    with open(save_path + args.source.split('.')[-2] + args.target.split('.')[-2] + '.pkl', 'wb') as handle:
        pkl.dump(pivot_adapter, handle)

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class PadSequence:
    
    def __init__(self, padding):
        self.padding = padding
    
    def __call__(self, batch):
        
        x = [s[0] for s in batch]
        x = pad_sequence(x, 
                         batch_first=True, 
                         padding_value=self.padding)

        y = [s[1] for s in batch]
        y = pad_sequence(y, 
                         batch_first=True, 
                         padding_value=self.padding)

        return x, y

class ParallelDataset(Dataset):
  
    def __init__(self, 
                 tokenizer,
                 src_path,
                 tgt_path):
    
        self.src_handle = h5py.File(src_path, 'r')
        self.tgt_handle = h5py.File(tgt_path, 'r')

        self.tokenizer = tokenizer

    def __len__(self):
        return self.src_handle.get('dataset').shape[0]

    def __getitem__(self, index):

        x = self.src_handle.get('dataset')[index]
        y = self.tgt_handle.get('dataset')[index]

        return torch.LongTensor(x), torch.LongTensor(y)

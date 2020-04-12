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
    
        src_handle = h5py.File(src_path, 'r')
        tgt_handle = h5py.File(tgt_path, 'r')

        self.src_sentences = np.array(src_handle.get('dataset'))
        self.tgt_sentences = np.array(tgt_handle.get('dataset'))

        self.tokenizer = tokenizer

    def __len__(self):
        return self.src_sentences.shape[0]

    def __getitem__(self, index):

        x = self.src_sentences[index].decode()
        y = self.tgt_sentences[index].decode()

        #tokenize into integer indices
        x = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))
        y = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(y))

        return torch.LongTensor(x), torch.LongTensor(y)

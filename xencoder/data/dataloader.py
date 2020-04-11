import torch
from torch.utils.data import Dataset, DataLoader
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
                 tgt_path,
                 mode='train'):
    
        src_handle = open(src_path, 'r')
        tgt_handle = open(tgt_path, 'r')

        self.src_sentences = []
        self.tgt_sentences = []

        for s, t in zip(src_handle.readlines(), tgt_handle.readlines()):
            self.src_sentences.append(s.strip())
            self.tgt_sentences.append(t.strip())
      
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, index):

        x = self.src_sentences[index]
        y = self.tgt_sentences[index]
 
        #tokenize into integer indices
        x = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))
        y = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(y))

        return torch.LongTensor(x), torch.LongTensor(y)

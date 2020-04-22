import argparse
import pandas as pd
import numpy as np
import h5py
from transformers import XLMRobertaTokenizer

def process_data(args):
       
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    df = pd.read_csv(args.source, chunksize=args.chunksize, sep='\t', header=None)
    
    #Creating HDF5 dataset
    hf = h5py.File(args.source + '.h5', 'w')
    dataset = None

    for chunk in df:

        sentences = []
 
        for sentence in chunk[0]:

            token_ids = np.array(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)))
            if len(token_ids) < args.max_length:
                sentences.append(token_ids)
        
        if dataset is None:
            dataset = hf.create_dataset('dataset',
                                        (len(sentences),),
                                        maxshape=(None,), 
                                        chunks=True, 
                                        dtype=h5py.special_dtype(vlen=np.dtype('int32')))
        else:
            dataset.resize(dataset.shape[0] + len(sentences), axis=0)  

        dataset[-len(sentences):] = np.array(sentences)

    hf.close()
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pre-Training the Pivot Adapter Network')

    parser.add_argument('--source', type=str, help='Path to source corpus')
    parser.add_argument('--chunksize', type=int, help='Chunk size')
    parser.add_argument('--save_path', type=str, help='Destination to store the adapter')
    parser.add_argument('--max_length', type=int, help='Maximum length of a sequence of tokens (not words)')

    args = parser.parse_args()

    process_data(args)

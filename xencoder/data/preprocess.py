import argparse
import pandas as pd
import numpy as np
import h5py

def get_tokenizer(args):

    if args.tokenizer == 'xlmr':

        from transformers import XLMRobertaTokenizer
        xlmrtokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

        def tokenizer(sentence):
            return xlmrtokenizer.convert_tokens_to_ids(xlmrtokenizer.tokenize(sentence))
        
        return tokenizer

    elif args.tokenizer == 'vanilla':
    
        from fairseq.data.encoders.subword_nmt_bpe import SubwordNMTBPE
        from fairseq.data.encoders.moses_tokenizer import MosesTokenizer
        from fairseq.data import Dictionary        
 
        class Args:

            def __init__(self):

                self.moses_source_lang = 'en'
                self.moses_target_lang = 'fr'
                self.moses_no_dash_splits = False
                self.moses_no_escape = False
                self.bpe_codes = './bpecodes'
                self.bpe_separator = '@@'

        def tokenizer(sentence):

            attrs = Args()
            
            tokenizer = MosesTokenizer(attrs)
            bpe = SubwordNMTBPE(attrs)
            dictionary = Dictionary.load('dict.en.txt')
            
            return dictionary.encode_line(bpe.encode(sentence))
           
        return tokenizer

def process_data(args):
    
    tokenizer = get_tokenizer(args)   

    df = pd.read_csv(args.source, chunksize=args.chunksize, sep='\t', header=None)
    
    #Creating HDF5 dataset
    hf = h5py.File(args.source + '.h5', 'w')
    dataset = None

    for chunk in df:

        sentences = []
 
        for sentence in chunk[0]:

            token_ids = np.array(tokenizer(sentence))
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
    parser.add_argument('--tokenizer', type=str, help='Tokenizer depending on XLMR vs Encoder in Vaswani et. al')

    args = parser.parse_args()

    process_data(args)

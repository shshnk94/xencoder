import argparse
import pandas as pd
import numpy as np
import h5py

def get_tokenizer(tok_type):

    if tok_type == 'xlmr':

        from transformers import XLMRobertaTokenizer
        xlmrtokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

        def tokenizer(sentence):
            return xlmrtokenizer.convert_tokens_to_ids(xlmrtokenizer.tokenize(sentence))
        
        return tokenizer

    elif tok_type == 'vanilla':
    
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
    
    src_tokenizer = get_tokenizer(args.src_tokenizer)   
    tgt_tokenizer = get_tokenizer(args.tgt_tokenizer)   

    src = pd.read_csv(args.source, chunksize=args.chunksize, sep='\t', header=None)
    tgt = pd.read_csv(args.target, chunksize=args.chunksize, sep='\t', header=None)
    
    #Creating HDF5 dataset
    src_hf = h5py.File(args.source + '.h5', 'w')
    tgt_hf = h5py.File(args.target + '.h5', 'w')

    src_dataset = None
    tgt_dataset = None

    for src_chunk, tgt_chunk in zip(src, tgt):

        src_sentences = []
        tgt_sentences = []

        for src_sentence, tgt_sentence in zip(src_chunk[0], tgt_chunk[0]):

            src_token_ids = np.array(src_tokenizer(src_sentence))
            tgt_token_ids = np.array(tgt_tokenizer(tgt_sentence))

            if len(src_token_ids) < args.src_max_length and len(tgt_token_ids) < args.tgt_max_length:
                src_sentences.append(src_token_ids)
                tgt_sentences.append(tgt_token_ids)
        
        if src_dataset is None and tgt_dataset is None:

            src_dataset = src_hf.create_dataset('dataset',
                                            (len(src_sentences),),
                                            maxshape=(None,), 
                                            chunks=True, 
                                            dtype=h5py.special_dtype(vlen=np.dtype('int32')))

            tgt_dataset = tgt_hf.create_dataset('dataset',
                                            (len(tgt_sentences),),
                                            maxshape=(None,), 
                                            chunks=True, 
                                            dtype=h5py.special_dtype(vlen=np.dtype('int32')))
        else:
            src_dataset.resize(src_dataset.shape[0] + len(src_sentences), axis=0)  
            tgt_dataset.resize(tgt_dataset.shape[0] + len(tgt_sentences), axis=0)  

        src_dataset[-len(src_sentences):] = np.array(src_sentences)
        tgt_dataset[-len(tgt_sentences):] = np.array(tgt_sentences)

    src_hf.close()
    tgt_hf.close()
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pre-Training the Pivot Adapter Network')

    parser.add_argument('--source', type=str, help='Path to source corpus')
    parser.add_argument('--target', type=str, help='Path to source corpus')
    parser.add_argument('--chunksize', type=int, help='Chunk size')
    parser.add_argument('--src_max_length', type=int, help='Maximum length of a sequence of tokens (not words) in source')
    parser.add_argument('--tgt_max_length', type=int, help='Maximum length of a sequence of tokens (not words) in target')
    parser.add_argument('--src_tokenizer', type=str, help='Tokenizer depending on XLMR vs Encoder in Vaswani et. al')
    parser.add_argument('--tgt_tokenizer', type=str, help='Tokenizer depending on XLMR vs Encoder in Vaswani et. al')

    args = parser.parse_args()

    process_data(args)

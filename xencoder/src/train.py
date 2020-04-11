from .config import Config
from .translation import TranslationModel
from ..utils.modules import flat_accuracy
from ..data.dataloader import ParallelDataset, PadSequence

import argparse
from time import time

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from transformers import XLMRobertaModel, XLMRobertaTokenizer

seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# To be used once config is set
# def train(config, model, train_loader, val_loader):

def train(model, train_loader, val_loader):

    training_loss_values = []
    validation_loss_values = []

    model = TranslationModel()
    for name, param in model.named_parameters():
    param.requires_grad = False if 'encoder' in name else True

    model.cuda()

    criterion = nn.MSELoss(reduction='sum')

    for epoch in range(4):

        print('======== Epoch ========', epoch)
	  
        total_loss = 0
		  
        for batch_no, batch in enumerate(train_loader):
            source = batch[0].to(device)
            target = batch[1].to(device)

            model.zero_grad()        

            source, target = model(source, target)
            loss = criterion(source, target)
            total_loss += loss.item()

			loss.backward()
		  
		avg_train_loss = total_loss / len(train_loader)
		training_loss_values.append(avg_train_loss)

		print("Average training loss: {0:.2f}".format(avg_train_loss))
		print("Running Validation...")

		model.eval()

		eval_loss = 0
		nb_eval_steps = 0

		for batch_no, batch in enumerate(val_loader):
			
			source = batch[0].to(device)
			target = batch[1].to(device)
			
			with torch.no_grad():        
				loss = criterion(source, target)
				eval_loss += loss
				nb_eval_steps += 1

		avg_valid_loss = eval_loss/nb_eval_steps
		validation_loss_values.append(avg_valid_loss)

		print("Average validation loss: {0:.2f}".format(avg_valid_loss))
		    
		print("Time taken by epoch: {0:.2f}".format(time() - start_time))

	return training_loss_values, validation_loss_values


# To be used once config is set
# def build_model(config):

def build_model():
    
    src_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tgt_tokenizer.bos_token = '<s>'
    tgt_tokenizer.eos_token = '</s>'
    
	#hidden_size and intermediate_size are both wrt all the attention heads. 
    #Should be divisible by num_attention_heads
    encoder_config = BertConfig(vocab_size=src_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12)

    decoder_config = BertConfig(vocab_size=tgt_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12,
                                is_decoder=True)

    #Create encoder and decoder embedding layers.
    encoder_embeddings = nn.Embedding(src_tokenizer.vocab_size, config.hidden_size, padding_idx=src_tokenizer.pad_token_id)
    decoder_embeddings = nn.Embedding(tgt_tokenizer.vocab_size, config.hidden_size, padding_idx=tgt_tokenizer.pad_token_id)

    encoder = BertModel(encoder_config)
    encoder.set_input_embeddings(encoder_embeddings)
    
    decoder = BertForMaskedLM(decoder_config)
    decoder.set_input_embeddings(decoder_embeddings)

    model = TranslationModel(encoder, decoder)

    return model, src_tokenizer, tgt_tokenizer

def run_experiment(config):

    model, src_tokenizer, tgt_tokenizer = build_model(config)
    
    pad_sequence = PadSequence(src_tokenizer.pad_token_id, tgt_tokenizer.pad_token_id) 
    train_loader = DataLoader(IndicDataset(src_tokenizer, tgt_tokenizer, config.data, True), 
                              batch_size=config.batch_size, 
                              shuffle=False, 
                              collate_fn=pad_sequence)
    eval_loader = DataLoader(IndicDataset(src_tokenizer, tgt_tokenizer, config.data, False), 
                             batch_size=config.eval_size, 
                             shuffle=False, 
                             collate_fn=pad_sequence)

    writer = SummaryWriter(config.output + 'logs/') 

    (training_loss_values, 
     validation_loss_values, 
     validation_accuracy_values) = train(config, model, train_loader, eval_loader, writer)
    
    return training_loss_values, validation_loss_values, validation_accuracy_values

"""
if __name__ == '__main__':
    hin_config = Config(epochs=40, 
                        batch_size=64, 
                        eval_size=16, 
                        vocab_size=25000, 
                        embed_dim=100, 
                        hidden_size=256, 
                        intermediate_size=512,
                        num_attention_heads=1, 
                        num_hidden_layers=1)
    run_experiment(hin_config)
"""

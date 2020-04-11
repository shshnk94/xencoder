import torch.nn as nn


class TranslationModel(nn.Module):

    def __init__(self):
        
        super(TranslationModel, self).__init__()
        self.encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.linear = nn.Linear(768, 768)

    def forward(self, src_tensors, tgt_tensors):

        src_tensors = self.encoder(src_tensors)[0].mean(axis=1)
        src_embeddings = self.linear(src_tensors.cuda())
        tgt_embeddings = self.encoder(tgt_tensors)[0].mean(axis=1)

    return src_embeddings.cuda(), tgt_embeddings.cuda()
   

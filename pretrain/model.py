
from cProfile import label
import os
import random
import numpy as np
import gc
import copy
from transformers import  T5ForConditionalGeneration
from transformers import T5Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyper import *
import logging
import torch
from utils import *
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss


class Mem_Model(torch.nn.Module):
    def __init__(self,model,hs=512):
        super(Mem_Model,self).__init__()
        self.model = model.cuda()
        for k in self.model.parameters():
            k.requires_grad=True
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.lm_head = self.model.lm_head
        self.mem =  nn.Parameter(torch.rand((1,hs),device='cuda',requires_grad=True))
    def forward(self,x,x_attn,labels,labels_attn):
        en_out = self.encoder(x,x_attn)[0]
        temp = self.mem.repeat(x.shape[0],1,1)
        en_out_new = torch.hstack((en_out,temp)) 
        decoder_input_ids = shift_right(labels)
        decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids,
                    attention_mask=labels_attn,
                    encoder_hidden_states=en_out_new
                )
        sequence_output  = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)
        loss_fct = CrossEntropyLoss(ignore_index=0)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        return lm_logits,loss
    # def generate(self):
    #     self.

        

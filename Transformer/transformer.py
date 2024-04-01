import torch
from torch import nn 

import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
import torch.nn as nn

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
    def forward(self):
        even_i = torch.arrange(0,self.d_model,2).float()
        denominator = torch.pow(10000,even_i/self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length,1)
        even_PE = torch.sin(position/denominator)
        odd_PE = torch.cos(position/denominator)
        stacked = torch.stack([even_PE,odd_PE],dim=2)
        PE = torch.flatten(stacked,start_dim=1,end_dim=2)
        return PE

'''
pe = PositionalEncoding(d_model=6,max_sequence_length=10)
pe.forward()
'''
      
class SentenceEmbedding(nn.Module):
    def __init__(self,max_sequence_length,d_model,language_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length=max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size,d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model,max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN=START_TOKEN
        self.END_TOKEN=END_TOKEN
        self.PADDING_TOKEN=PADDING_TOKEN

    def batch_tokenizer(self,batch,start_token,end_token):
        def tokenize(sentence,start_token,end_token):
            sentence_word_indicies=[self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0,self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies),self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized=[]
        for sentence_num in range(len(batch)):
            tokenized.append(tokenize(batch[sentence_num],start_token,end_token))
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())

    def forward(self,x,start_token,end_token):
        x=self.batch_tokenize(x,start_token,end_token)
        x=self.embedding(x)
        pos=self.position_encoder().to(get_device())
        x=self.dropout(x+pos)
        return x
        

class SequenceDecoder(nn.Sequential):
    def forword(self,*inputs):
        x,y,mask = inputs
        for module in self.modules.values():
            y = module(x,y,mask)#30x200x512
        return y


class LayerNormalization(nn.Module):
    def __init__(self,parameters_shape,eps=1e5):
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma=nn.Parameter(torch.ones(parameters_shape))
        self.alpha=nn.Parameter(torch.zeros(parameters_shape))

    def forward(self,inputs):
        dims=[-(i+1) for i in range(len(self.parameters_shape))]
        mean=inputs.mean(dim=dims,keepdim=True)
        variance=((inputs-mean)**2).mean(dim=dims,keepdim=True)
        standard_dev=(variance+self.eps).sqrt()
        z=(inputs-mean)/standard_dev
        out=self.gamma*z+self.alpha
        return out
'''
batch_size=3
sentence_length=5
embedding_dim=8
inputs=torch.randn(sentence_length,batch_size,embedding_dim)

#print(f"{inputs.size()}\n{inputs}")
layer=LayerNormalization(inputs.size()[-2:])
out=layer.forward(inputs)
print(out)
'''
class EncoderLayer(nn.Module):
    def __init__(self, d_model,ffn_hidden,num_heads,drop_prob):
        super(EncoderLayer,self).__init__()
        self.attention=MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.norm1=LayerNormalization(parameters_shape=[d_model])
        self.drop1=nn.Dropout(p=drop_prob)
        self.ffn=PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm2=LayerNormalization(parameters_shape=[d_model])
        self.drop2=nn.Dropout(p=drop_prob)
    
    def forward(self,x):
        residual_x=x
        x=self.attention(x,mask=None)
        x=self.drop1(x)
        x=self.norm1(x+residual_x)
        x=self.ffn(x)
        x=self.drop2(x)
        x=self.norm2(x+residual_x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x

class SequentialEncoder(nn.Sequential):
    def forward(self,*inputs):
        x,sa_mask=inputs
        for module in self._modules.values():
            x=module(x,sa_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
        super(DecoderLayer,self).__init__()
        self.self_attention=MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.norm=LayerNormalization(parameters_shape=[d_model])
        self.drop=nn.Dropout(p=drop_prob)
        self.encdecattent=MultiHeadCrossAttention(d_model=d_model,num_heads=num_heads)
        self.ffn=PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        

    def forward(self,x,y,dmask):
        resy=y
        y=self.self_attention(y,mask=dmask)
        y=self.drop(y)
        y=self.norm(y+resy)

        resy=y
        y=self.encdecattent(x,y,mask=dmask)
        y=self.drop(y)
        y=self.norm(y+resy)

        resy=y
        y=self.ffn(y)
        y=self.drop(y)
        y=self.norm(y)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size(-1)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    attention = F.softmax(scaled, dim=-1)
    if mask is not None:
        attention = attention.masked_fill(mask == float('-inf'), 0.0)
    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3)
        out = self.linear_layer(values.reshape(batch_size, sequence_length, -1))
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask):
        batch_size, sequence_length, d_model = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out


class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y


class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_sequence_length, 
                kn_vocab_size,
                english_to_index,
                kannada_to_index,
                START_TOKEN, 
                END_TOKEN, 
                PADDING_TOKEN
                ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, kn_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make this true
                dec_end_token=False): # x, y are batch of sentences
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out
    
    
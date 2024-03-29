import torch
from torch import nn 
from Sandeep import MultiHeadAttention,MultiHeadCrossAttention,scaled_dot_product,PositionwiseFeedForward
from Varshini import PositionalEncoding,SentenceEmbedding,SequenceDecoder
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
    def __init__(self, d_model,ffn_hidden,num_heads,drop_prob,num_layers):
        super(Encoder,self).__init__()
        self.layers=nn.Sequential(*[EncoderLayer(d_model,ffn_hidden,num_heads,drop_prob)] for _ in range(num_layers))
    
    def forward(self,x):
        x=self.layers(x)
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




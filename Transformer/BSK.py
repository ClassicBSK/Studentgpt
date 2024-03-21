import torch
from torch import nn 
class LayerNormalization():
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
class EncoderLayer():
    values=0

class SequentialEncoder():
    values=0

class DecoderLayer():
    values=0


import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def__init__(self, d_model, max_sequence_length):
        super()__init__()
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
      
class SentenceEmbedding():
    value=0

class SequenceDecoder():
    def forword(self,*inputs):
        x,y,mask = inputs
        for module in self_modules.values():
            y = module(x,y,mask)#30x200x512
    return y

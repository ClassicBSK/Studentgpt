from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import checkpoint

from transformers import AdamW

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import config

def collate_fn(batch):
    return {
        'text':torch.stack([x['text'] for x in batch]),
        'summary':torch.stack([x['summary'] for x in batch]),
        'text_input_ids':torch.tensor([x['text_input_ids'] for x in batch]),
        'text_attention_mask':torch.tensor([x['text_attention_mask'] for x in batch]),
        'labels':torch.tensor([x['labels'] for x in batch]),
        'labels_attention_mask':torch.tensor([x['labels_attention_mask'] for x in batch])
    }

#creating torch dagaset 
class SummaryDataset(Dataset):
    def __init__(self,
                 data:pd.DataFrame,
                 tokenizer,
                 max_text_len:int = config.text_max_len,
                 sum_max_len:int = config.sum_max_len):
        
        self.tokenizer=tokenizer
        self.data=data
        self.max_text_len=max_text_len
        self.sum_max_len=sum_max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int):
        record=self.data.iloc[index]

        text=record['text']

        text_sequences=self.tokenizer(text,
                                      max_length=self.max_text_len,
                                      padding="longest",
                                      truncation=True,
                                      return_attention_mask=True,
                                      add_special_tokens=True,
                                      return_tensors='pt'
                                      )
        
        summary=record['summary']
        summary_sequences=self.tokenizer(summary,
                                        max_length=self.sum_max_len,
                                        padding="longest",
                                        truncation=True,
                                        return_attention_mask=True,
                                        add_special_tokens=True,
                                        return_tensors='pt')
        
        labels=summary_sequences['input_ids']
        labels[labels==0]=-100
        return {
            'text':text,
            'summary':summary,
            'text_input_ids':text_sequences['input_ids'].flatten(),
            'text_attention_mask':text_sequences['attention_mask'].flatten(),
            'labels':labels.flatten(),
            'labels_attention_mask':summary_sequences['attention_mask'].flatten()
        }
    

#creating pl lightning module
class SummaryModule(pl.LightningDataModule):
    def __init__(self,
                 train_df:pd.DataFrame,
                 val_df:pd.DataFrame,
                 tokenizer,
                 batch_size:int=config.batch_size,
                 text_max_len:int =config.text_max_len,
                 sum_max_len:int =config.sum_max_len):
        super().__init__()

        self.train_df=train_df
        self.val_df=val_df

        self.tokenizer=tokenizer
        self.batch_size=batch_size

        self.text_max_len=text_max_len
        self.sum_max_len=sum_max_len

    def setup(self,stage=None):

        self.train_dataset=SummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_len,
            self.sum_max_len
        )
        
        self.val_dataset=SummaryDataset(
            self.val_df,
            self.tokenizer,
            self.text_max_len,
            self.sum_max_len
        )
        
    def train_dataloader(self):
        #dataloader for training set
        return DataLoader(self.train_dataset,
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=config.num_workers,
                          collate_fn=collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=config.batch_size,
                          shuffle=False,
                          num_workers=config.num_workers,
                          collate_fn=collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=config.batch_size,
                          shuffle=False,
                          num_workers=config.num_workers,
                          collate_fn=collate_fn)
    

class SummaryModel(pl.LightningModule):
    def __init__(self,model=None):
        
        super().__init__()
        self.model=model

    def forward(self,input_ids,attention_mask,decoder_attention_mask,labels=None):
        output=self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels 
        )
        
        return output.loss,output.logits
    
    def checkType(self,input_ids,attention_mask,labels,labels_attention_mask,mode="training"):
        if type(input_ids)!=torch.Tensor or type(attention_mask)!=torch.Tensor or type(labels)!=torch.Tensor or type(input_ids)!=torch.Tensor:
            raise ValueError()

    def training_step(self,batch,batch_idx):
        input_ids=batch['text_input_ids']
        attention_mask=batch['text_attention_mask']
        labels=batch['labels']
        labels_attention_mask=batch['labels_attention_mask']

        loss,output=self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("train loss",loss,prog_bar=True,logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        #self.checkType(input_ids, attention_mask, labels, labels_attention_mask, mode = "val")

        loss, outputs = self(input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            decoder_attention_mask = labels_attention_mask
            )
        
        self.log("val_loss", loss, prog_bar = True, logger = True)

        return loss
    
    def test_step(self, batch,batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        #self.checkType(input_ids, attention_mask, labels, labels_attention_mask, mode = "val")

        loss, outputs = self(input_ids = input_ids,
        attention_mask = attention_mask,
        labels = labels,
        decoder_attention_mask = labels_attention_mask
        )
        
        self.log("test_loss", loss, prog_bar = True, logger = True)

        return loss
    
    def configure_optimizers(self):
        
        return AdamW(self.parameters(),lr=config.learning_rate)
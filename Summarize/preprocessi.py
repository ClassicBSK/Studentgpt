from SummaryModules import SummaryDataset,SummaryModel,SummaryModule
from config import config

import pandas as pd
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger




class Preprocessing:

    def __init__(self):
        raw_data=pd.read_csv("./news_summary.csv",encoding='latin-1')
        data=raw_data[['ctext','text']]
        data.columns=['text','summary']
        self.data=data.dropna()
    
        self.train_df,self.val_df=train_test_split(self.data,test_size=0.1)
        self.get_data_module()
        # print(train_df.shape)
        # print(val_df.shape)

        #print(data.head())
        #t5_data_module=SummaryModule(train_df)

    def get_data_module(self):
        self.t5_data_module=SummaryModule(train_df= self.train_df, val_df=self.val_df,tokenizer=config.t5_tokenizer,batch_size=config.batch_size)
        

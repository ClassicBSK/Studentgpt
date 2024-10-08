from SummaryModules import SummaryDataset,SummaryModel,SummaryModule
from config import config
from preprocessi import Preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


raw_data=pd.read_csv("./news_summary.csv",encoding='latin-1')

data=raw_data[['ctext','text']]

data.columns=['text','summary']

data=data.dropna()

train_df,val_df=train_test_split(data,test_size=0.1)
print(train_df.shape)
print(val_df.shape)

#print(data.head())
#t5_data_module=SummaryModule(train_df)

t5_data_module=SummaryModule(train_df=train_df,val_df=val_df,tokenizer=config.t5_tokenizer,batch_size=config.batch_size)

class Summarize:
        def __init__(self):
                pass

        def train(self,preprocessing:Preprocessing):
            t5_model=SummaryModel(config.t5_pretrained_model)

            t5_checkpoint_callback=ModelCheckpoint(
                dirpath="t5-checkpoints",
                filename="t5-best-checkpoint",
                save_top_k=1,
                verbose=True,
                monitor="val_loss",
                mode="min",
            )

            t5_logger=TensorBoardLogger("t5_logs",name='t5-news-summary')

            t5_trainer=pl.Trainer(
                logger=t5_logger,
                callbacks=t5_checkpoint_callback,
                max_epochs=config.n_epochs,
                enable_progress_bar=True
            )

            t5_trainer.fit(t5_model,preprocessing.t5_data_module)

            return t5_model


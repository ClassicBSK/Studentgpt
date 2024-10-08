from preprocessi import Preprocessing
from summariz import Summarize

if __name__=='__main__':

    preprocessing=Preprocessing()

    summarizer=Summarize()
    t5_model=summarizer.train(preprocessing=preprocessing)

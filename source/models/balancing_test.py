import pandas as pd

import sys
sys.path.append("..")
sys.path.append("../..")

import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

from features.vectorizer import vectorize
from features.balancer import balance

def balancing(cfg: GlobalConfig):
    print ('====== Balancing started =======')
    print ('================================')

    df = pd.read_csv(cfg.paths.processed)
    y = df[cfg.dataset.area]
    x = df[cfg.dataset.preprocessed]

    x = vectorize(cfg,x)

    x,y = balance(cfg,x,y)
    
    print ('================================')
    print ('======= Balancing ended ========')

if __name__ == '__main__':
    balancing()
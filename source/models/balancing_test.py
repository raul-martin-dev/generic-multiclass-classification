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
from features.spliter import split
from features.balancer import balance
from features.visualizer import bar_viusualize

def balancing(cfg: GlobalConfig):
    print ('====== Balancing started =======')
    print ('================================')

    df = pd.read_csv(cfg.paths.processed)
    y = df[cfg.dataset.area]
    x = df[cfg.dataset.preprocessed]

    # initial visualization
    ammount = df[cfg.dataset.area]
    bar_viusualize(ammount,cfg.dataset.area,cfg.dataset.description)

    x = vectorize(cfg,x)

    X_train, X_test, Y_train, Y_test = split(cfg,x,y)

    X_train,Y_train = balance(cfg,X_train,Y_train)

    # balanced data visualization
    bar_viusualize(Y_train,cfg.dataset.area,cfg.dataset.description)
    
    print ('================================')
    print ('======= Balancing ended ========')

if __name__ == '__main__':
    balancing()
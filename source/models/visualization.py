import pandas as pd

import sys
sys.path.append("..")
sys.path.append("../..")

import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

from features.visualizer import bar_viusualize

def visualization(cfg: GlobalConfig):
    print ('==== Visualization started =====')
    print ('================================')

    df = pd.read_csv(cfg.paths.processed)
    ammount = df[cfg.dataset.area]
    bar_viusualize(ammount,cfg.dataset.area,cfg.dataset.description)
    
    print ('================================')
    print ('===== Visualization ended ======')

if __name__ == '__main__':
    visualization()
import pandas as pd

import sys
sys.path.append("..")
sys.path.append("../..")

import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

def clean(cfg: GlobalConfig):
    raw_data = pd.read_csv(cfg.paths.raw)
    df = pd.DataFrame(raw_data)
    # Get only desired fields
    clean_data = df.loc[:, [cfg.dataset.area,cfg.dataset.description]]
    # Clean NaN
    clean_data = clean_data.dropna()
    # Clean No comments
    clean_data = clean_data[clean_data[cfg.dataset.description]!= "No comments"]
    # Clean duplicate values
    clean_data = clean_data.drop_duplicates()
    

    clean_data.to_csv(cfg.paths.clean)

if __name__ == '__main__':
    clean()

    
    
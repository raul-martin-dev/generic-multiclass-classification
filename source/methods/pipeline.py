import sys
sys.path.append("..")
sys.path.append("../..")

import pandas as pd

# data
from data.clean import clean
from data.preprocess import preprocess

#features
from features.visualizer import bar_viusualize

# models
from models.visualization import visualization
from models.balancing_test import balancing

import spacy
spanish = spacy.load('es_core_news_sm')

import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

@hydra.main(version_base=None, config_path='../../config', config_name='config')

def main(cfg: GlobalConfig):
    if cfg.pipeline.cleaning :
        clean(cfg)
    if cfg.pipeline.preprocessing :
        preprocess(cfg)

    df = pd.read_csv(cfg.paths.processed)
    y = df[cfg.dataset.area]
    x = df[cfg.dataset.preprocessed]

    if cfg.visualization.graph.initial:
        print ('\n==== initial visualization started =====')
        print ('========================================\n')

        ammount = df[cfg.dataset.area]
        bar_viusualize(ammount,cfg.dataset.area,cfg.dataset.description)
        
        print ('\n========================================')
        print ('===== initial visualization ended ======\n')

    # models
    if cfg.models.model == 'visualization' :
        visualization(cfg)
    if cfg.models.model == 'balancing_test' :
        balancing(cfg)

if __name__ == '__main__':
    sys.argv.append('hydra/job_logging=disabled')
    main()
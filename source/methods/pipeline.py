import sys
sys.path.append("..")
sys.path.append("../..")

from data.clean import clean
from data.preprocess import preprocess

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

    # models
    if cfg.models.model == 'visualization' :
        visualization(cfg)
    if cfg.models.model == 'balancing_test' :
        balancing(cfg)

if __name__ == '__main__':
    sys.argv.append('hydra/job_logging=disabled')
    main()
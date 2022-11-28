import sys
sys.path.append("..")
sys.path.append("../..")

from data.clean import clean
from data.preprocess import preprocess
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

if __name__ == '__main__':
    sys.argv.append('hydra/job_logging=disabled')
    main()
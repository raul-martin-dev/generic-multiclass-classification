from Preln.preprocessing import Preprocessing

import sys
sys.path.append("..")
sys.path.append("../..")

import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

def preprocess(cfg: GlobalConfig):
    print("\n> Preprocessing the dataset")

    preprocessor = Preprocessing( date_format = cfg.preprocessing.date_format
                                ,accents = cfg.preprocessing.accents
                                ,lowercasing = cfg.preprocessing.lowercasing
                                ,numbers = cfg.preprocessing.numbers
                                ,debug = cfg.preprocessing.debug
                                ,punctuation = cfg.preprocessing.punctuation
                                ,stopwords = cfg.preprocessing.stopwords
                                ,tokenizer = cfg.preprocessing.tokenizer)
    # preprocessor = Preprocessing()

    test = preprocessor.filePipeline(cfg.paths.clean, cfg.dataset.description)
    preprocessor.write(test,cfg.paths.clean,cfg.paths.processed) # pendiente de cambiar con preln (añadir el nombre de la columna texto preprocesado)
    
    print('\033[92m'+"Preprocessing ended successfully\n"+'\033[0m')
                            
if __name__ == '__main__':
    sys.argv.append('hydra/job_logging=disabled')
    preprocess()
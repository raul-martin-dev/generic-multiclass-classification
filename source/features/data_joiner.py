import pandas as pd

import sys
sys.path.append("..")
sys.path.append("../..")

import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

@hydra.main(version_base=None, config_path='../../config', config_name='config')

# def combine(cfg: GlobalConfig):
#     # with open(cfg.paths.raw_folder + '2016.csv') as f:
#     #     print(f)
#     dataset1 = pd.read_csv(cfg.paths.raw_folder + '2016.csv', on_bad_lines='skip', sep=';')
#     # dataset1.to_csv(cfg.paths.raw_folder + '2016_2.csv', encoding='utf-8', index=False)
#     dataset1 = pd.DataFrame(dataset1)
#     print (dataset1.loc[:, ['Tipo de accidente', 'Descripción del accidente']])

def provitional(cfg: GlobalConfig):
    dataset1 = pd.read_csv(cfg.paths.raw_folder + '2016.csv', on_bad_lines='skip', sep=';')
    dataset1.to_csv(cfg.paths.raw, encoding='utf-8', index=False)

if __name__ == '__main__':
    provitional()
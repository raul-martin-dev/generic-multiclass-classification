import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

from sklearn.model_selection import train_test_split

def split(cfg: GlobalConfig,x,y):
    X_train,X_test, Y_train,Y_test = train_test_split(x,y,test_size=cfg.splitting.test_size, random_state=25)
    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    split()
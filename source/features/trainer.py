import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import time
from datetime import date

import os

import pickle

import shutil

def train(cfg: GlobalConfig,X_train,Y_train,model):
    if model == "logistic":
        trained_model = LogisticRegression(C=cfg.models.logistic.C, 
                                            penalty=cfg.models.logistic.penalty, 
                                            max_iter=cfg.models.logistic.max_iter, 
                                            multi_class=cfg.models.logistic.multi_class
                                            ).fit(X_train, Y_train)
    elif model == 'mlp':
        trained_model = MLPClassifier(hidden_layer_sizes=cfg.models.mlp.hidden_layer_sizes, 
                                        max_iter=cfg.models.mlp.max_iter, 
                                        activation=cfg.models.mlp.activation, 
                                        solver=cfg.models.mlp.solver, 
                                        random_state=cfg.models.mlp.random_state, 
                                        learning_rate=cfg.models.mlp.learning_rate, 
                                        learning_rate_init=cfg.models.mlp.learning_rate_init).fit(X_train, Y_train)
    if cfg.models.store:
        if cfg.debuggin.warnings:
            print("\n" + '\033[93m' + "> WARNING:" + '\033[0m')
            print('\033[93m' + "Storing model is enabled. This could take some disk space. If the model doesn't have to be stored, you can disable the 'store' feature in the config file\n" + '\033[0m')
        day_folder = date.today().strftime("%m-%d-%Y")
        t = time.localtime()
        hour = time.strftime("%H_%M_%S", t)
        filename = cfg.dataset.alias + "-" + model + "_model-" + hour
        path = day_folder + "/" + cfg.dataset.alias + "/" + filename + "/"
        file = filename + ".sav"
        config_file = cfg.paths.config_file

        path = os.path.join(cfg.paths.store_models, path)
        os.makedirs(path)

        filepath = path + file
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

        shutil.copy(config_file, path + "config_used.txt")

    return trained_model

if __name__ == '__main__':
    train()
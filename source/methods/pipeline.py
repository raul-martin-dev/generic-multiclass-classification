import sys
sys.path.append("..")
sys.path.append("../..")

import pandas as pd

# data
from data.clean import clean
from data.preprocess import preprocess

#features
from features.visualizer import bar_viusualize
from features.visualizer import confusion_matrix_viusualize
from features.visualizer import metrics_visualize
from features.vectorizer import vectorize
from features.splitter import split
from features.balancer import balance
from features.trainer import train
from features.tester import test

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
    
    if cfg.pipeline.data_skip==False :
        if cfg.pipeline.cleaning :
            clean(cfg)

        if cfg.pipeline.preprocessing :
            preprocess(cfg)

        if cfg.pipeline.preprocessing :
            df = pd.read_csv(cfg.paths.processed)
        else :
            if cfg.pipeline.cleaning :
                df = pd.read_csv(cfg.paths.clean)
            else:
                df = pd.read_csv(cfg.paths.raw)
    else:
        df = pd.read_csv(cfg.paths.processed)

    y = df[cfg.dataset.area]
    x = df[cfg.dataset.preprocessed]

    if cfg.visualization.graph.initial:
        print ('\n==== initial visualization started ====')
        print ('=======================================\n')

        ammount = df[cfg.dataset.area]
        bar_viusualize(ammount,cfg.dataset.area,cfg.dataset.description)
        
        print ('\n=======================================')
        print ('===== initial visualization ended =====\n')
    
    if cfg.pipeline.vectorizing :
        x = vectorize(cfg,x)

    if cfg.pipeline.splitting :
        X_train, X_test, Y_train, Y_test = split(cfg,x,y)

    if cfg.pipeline.balancing :
        X_train,Y_train = balance(cfg,X_train,Y_train)
        if cfg.visualization.graph.post_balance:
            print ('\n===== post-balance visualization started =====')
            print ('==============================================\n')

            bar_viusualize(Y_train,cfg.dataset.area,cfg.dataset.description)
            
            print ('\n============================================')
            print ('===== post-balance visualization ended =====\n')

    if cfg.pipeline.training :
        model = train(cfg,X_train,Y_train,model=cfg.models.model)

    if cfg.pipeline.testing :
        predictions = test(model,X_test)
        if cfg.visualization.matrix:
            print ('\n===== confusion matrix visualization started =====')
            print ('==================================================\n')

            confusion_matrix_viusualize(Y_test, predictions)
            
            print ('\n================================================')
            print ('===== confusion matrix visualization ended =====\n')
        if cfg.visualization.metrics:
            print ('\n===== metrics visualization started =====')
            print ('=========================================\n')

            metrics_visualize(cfg, Y_test, predictions)
            
            print ('\n=======================================')
            print ('===== metrics visualization ended =====\n')

if __name__ == '__main__':
    sys.argv.append('hydra/job_logging=disabled')
    main()
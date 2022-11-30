import pandas as pd

import sys
sys.path.append("..")
sys.path.append("../..")

import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

from features.vectorizer import vectorize
from features.spliter import split
from features.balancer import balance
from features.visualizer import bar_viusualize

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def balancing(cfg: GlobalConfig):
    print ('====== Balancing started =======')
    print ('================================')

    df = pd.read_csv(cfg.paths.processed)
    y = df[cfg.dataset.area]
    x = df[cfg.dataset.preprocessed]

    # initial visualization
    ammount = df[cfg.dataset.area]
    bar_viusualize(ammount,cfg.dataset.area,cfg.dataset.description)

    x = vectorize(cfg,x)

    X_train, X_test, Y_train, Y_test = split(cfg,x,y)

    # X_train,Y_train = balance(cfg,X_train,Y_train)

    # balanced data visualization
    # bar_viusualize(Y_train,cfg.dataset.area,cfg.dataset.description)

    model = LogisticRegression(C=1000, penalty='l2', max_iter=10000, multi_class="ovr").fit(X_train, Y_train)

    # Predicciones
    predictions = model.predict(X_test)

    # Métricas y visualizaciones
    cm_bow = metrics.confusion_matrix(Y_test, predictions)

    class_label = Y_test.unique()
    df_cm = pd.DataFrame(cm_bow, index = class_label, columns = class_label)

    sns.heatmap(df_cm, annot = True, fmt = 'd')
    plt.title('Matriz de confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.show()
            
    print(f'  - El porcentaje de acierto es del: {np.round(metrics.accuracy_score(Y_test, predictions), 2) * 100}%')

    
    print ('================================')
    print ('======= Balancing ended ========')

if __name__ == '__main__':
    balancing()
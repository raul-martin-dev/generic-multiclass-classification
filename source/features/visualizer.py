import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import pandas as pd
import numpy as np

def bar_viusualize(ammount,area,description):
    print("\n> Generating bar graph")
    count = ammount.value_counts()
    count.plot.bar()
    plt.ylabel(description)
    plt.xlabel(area)
    plt.show()
    print('\033[92m'+"Bar graph visualization ended successfully\n"+'\033[0m')

def confusion_matrix_viusualize(Y_test,predictions):
    print("\n> Generating confusion matrix")
    cm_bow = metrics.confusion_matrix(Y_test, predictions)

    class_label = Y_test.unique()
    df_cm = pd.DataFrame(cm_bow, index = class_label, columns = class_label)

    sns.heatmap(df_cm, annot = True, fmt = 'd')
    plt.title('Confusion matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Reality')
    plt.show()
    print('\033[92m'+"Confusion matrix visualization ended successfully\n"+'\033[0m')

def metrics_visualize(cfg: GlobalConfig,Y_test, predictions):
    print("\n> Generating metrics")

    if cfg.testing.metrics.precision:
        print(f'  - Precision: {np.round(metrics.precision_score(Y_test, predictions,average="weighted"), 2) * 100}%')
    if cfg.testing.metrics.recall:
        print(f'  - Recall : {np.round(metrics.recall_score(Y_test, predictions,average="weighted"), 2) * 100}%')
    if cfg.testing.metrics.f1:
        print(f'  - F1: {np.round(metrics.f1_score(Y_test, predictions,average="weighted"), 2) * 100}%')
    if cfg.testing.metrics.accuracy:
        print(f'  - Accuracy: {np.round(metrics.accuracy_score(Y_test, predictions), 2) * 100}%')

    print('\033[92m'+'\n'+"Mectrics visualization ended successfully\n"+'\033[0m')
    

if __name__ == '__main__':
    bar_viusualize()
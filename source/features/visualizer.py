import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import pandas as pd

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

if __name__ == '__main__':
    bar_viusualize()
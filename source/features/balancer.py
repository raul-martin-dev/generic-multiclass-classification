import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler

def balance(cfg: GlobalConfig,x,y):
    if cfg.balancing.extra:
        print('\033[96m'+"\n> Extra mode detected:" +'\033[0m' + "Balancing the dataset (extra initial random oversampling) ...")
        dictionary = small_class_locator(y)
        resample = RandomOverSampler(sampling_strategy=dictionary)
        x, y = resample.fit_resample(x, y)
        print('\033[92m'+"Extra balancing ended successfully"+'\033[0m')

    if cfg.balancing.mode == "SMOTE":
        print("\n> Balancing the dataset (SMOTE strategy) ...")
        try:
            resample = SMOTE()
            x, y = resample.fit_resample(x, y)
        except Exception as e:
            # bcolors: https://stackoverflow.com/a/287944
            if type(e).__name__ == "ValueError":
                print("\n" + '\033[91m' + "> CRITICAL ERROR:" + '\033[0m')
                print('\033[91m' + "Something went wrong during the balancing. The 'SMOTE' mode is not supported for datasets with clases containing less than 6 samples. Toggling the 'extra' feature on the config file will randomly oversample those classes to 6 samples fixing this error.\n" + '\033[0m')
            else:
                print("\n" + '\033[91m' + "> CRITICAL ERROR:" + '\033[0m')
                print('\033[91m' + "Something went wrong during the balancing. Unknown Error: " + type(e).__name__ + '\033[0m')
        else:
            print('\033[92m'+"Balancing ended successfully\n"+'\033[0m')

    if cfg.balancing.mode == "TomekLinks":
        print("\n> Balancing the dataset (SMOTETomek, TomekLinks strategy) ...")
        try:
            resample = SMOTETomek(tomek=TomekLinks())
            x, y = resample.fit_resample(x, y)
        except Exception as e:
            if type(e).__name__ == "ValueError":
                print("\n" + '\033[91m' + "> CRITICAL ERROR:" + '\033[0m')
                print('\033[91m' + "Something went wrong during the balancing. The 'TomekLinks' mode is not supported for datasets with clases containing less than 6 samples. Toggling the 'extra' feature on the config file will randomly oversample those classes to 6 samples fixing this error.\n" + '\033[0m')
            else:
                print("\n" + '\033[91m' + "> CRITICAL ERROR:" + '\033[0m')
                print('\033[91m' + "Something went wrong during the balancing. Unknown Error: " + type(e).__name__ + '\033[0m')
        else:
            print('\033[92m'+"Balancing ended successfully\n"+'\033[0m')

    if cfg.balancing.mode == "SMOTEENN":
        print("\n> Balancing the dataset (SMOTEEN strategy)")
        try:
            resample = SMOTEENN()
            x, y = resample.fit_resample(x, y)
        except Exception as e:
            if type(e).__name__ == "ValueError":
                print("\n" + '\033[91m' + "> CRITICAL ERROR:" + '\033[0m')
                print('\033[91m' + "Something went wrong during the balancing. The 'SMOTEENN' mode is not supported for datasets with clases containing less than 6 samples. Toggling the 'extra' feature on the config file will randomly oversample those classes to 6 samples fixing this error.\n" + '\033[0m')
            else:
                print("\n" + '\033[91m' + "> CRITICAL ERROR:" + '\033[0m')
                print('\033[91m' + "Something went wrong during the balancing. Unknown Error: " + type(e).__name__ + '\033[0m')
        else:
            print('\033[92m'+"Balancing ended successfully\n"+'\033[0m')

    if cfg.balancing.mode == "Random":
        print("\n> Balancing the dataset (Random Oversampling strategy)")
        try:
            resample = RandomOverSampler()
            x, y = resample.fit_resample(x, y)
            if cfg.balancing.extra and cfg.debugging.warnings:
                    print("\n" + '\033[93m' + "> WARNING:" + '\033[0m')
                    print('\033[93m' + "The 'extra' feature is toggled on the config file. This feature is not necessary when using 'Random' mode\n" + '\033[0m')
        except Exception as e:
            print("\n" + '\033[91m' + "> CRITICAL ERROR:" + '\033[0m')
            print('\033[91m' + "Something went wrong during the balancing. Unknown Error: " + type(e).__name__ + '\033[0m')
        else:
            print('\033[92m'+"Balancing ended successfully\n"+'\033[0m')

    return x,y

def small_class_locator(y):
    # this method makes a dictionary out of the classes
    # that have less than 6 samples in order to use it as a
    # sampling strategy for an inital random oversampling
    # making the dataset suitable for the rest of the methods
    dict = {}
    serie = y.value_counts().iteritems()
    class_number = 0
    for key, value in serie:
        class_number += 1
        if value <= 5 :
            dict.update({key : 6})
    return dict

if __name__ == '__main__':
    balance()
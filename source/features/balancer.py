import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

def balance(cfg: GlobalConfig,x,y):
    if cfg.balancing.mode == "TomekLinks":
        print("\n> Balancing the dataset (SMOTETomek, TomekLinks strategy) ...")
        try:
            resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
            x, y = resample.fit_resample(x, y)
        except Exception as e:
            # bcolors: https://stackoverflow.com/a/287944
            if type(e).__name__ == "ValueError":
                print("\n" + '\033[91m' + "> CRITICAL ERROR:" + '\033[0m')
                print('\033[91m' + "Something went wrong during the balancing. The 'TomekLinks' mode is not supported for datasets with clases containing less than 5 samples. Try using 'TomekLinks-extra' mode instead.\n" + '\033[0m')
            else:
                print("\n" + '\033[91m' + "> CRITICAL ERROR:" + '\033[0m')
                print('\033[91m' + "Something went wrong during the balancing. Unknown Error: " + type(e).__name__ + '\033[0m')
        
    if cfg.balancing.mode == "TomekLinks-extra":
        print("\n> Balancing the dataset (SMOTETomek, TomekLinks strategy + ...)")
        # developing
        print ("x", x)
        print ("y", y)
        #print (y.value_counts())
    
    return x,y
    

if __name__ == '__main__':
    balance()
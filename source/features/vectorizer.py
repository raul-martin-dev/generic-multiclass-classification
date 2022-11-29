import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def vectorize(cfg: GlobalConfig, text_column):
    if cfg.vectorizing.mode == "tf-idf":
        bow_converter = TfidfVectorizer()
        text_column = bow_converter.fit_transform(text_column)
    if cfg.vectorizing.mode == "count":
        bow_converter = CountVectorizer(tokenizer=lambda doc: doc)
        text_column = bow_converter.fit_transform(text_column)
    return text_column
    

if __name__ == '__main__':
    # sample #= columna de un df
    # test = vectorize(sample)
    # print (test)
    vectorize()
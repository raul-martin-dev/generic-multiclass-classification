from dataclasses import dataclass
from distutils.command.clean import clean

@dataclass
class Paths:
    processed: str
    clean: str
    raw: str
    test: str
    test_exit: str

@dataclass
class Cleaning:
    area: str
    description: str
    
@dataclass
class Preprocessing:
    lw: bool
    pc: bool
    st: bool
    tk: bool

@dataclass
class Show:
    matrix: bool
    graphs: bool
    study: bool
    
@dataclass
class Models:
    model: str

@dataclass
class Pipeline:
    date : bool
    date_format : bool
    accents : bool
    lowercasing : bool
    numbers : bool
    debug : bool
    punctuation : bool
    stopwords : bool
    tokenizer : bool

@dataclass
class GlobalConfig:
    paths: Paths
    cleaning: Cleaning
    preprocessing: Preprocessing
    show: Show
    models: Models
    pipeline: Pipeline
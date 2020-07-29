import json

from easydict import EasyDict as edict


class Config:

    __init__ = lambda self: None

    def __new__(cls):
        if not hasattr(cls, '__config'):
            with open('config.json', 'r') as file:
                cls.__config = edict(json.load(file))
        return cls.__config


#   "TRAIN_FOLDERS" : ["2018", "2017", "2015", "2014", "2013", "2011", "2009"],
#   "TEST_FOLDERS" : ["2008", "2006", "2004"],

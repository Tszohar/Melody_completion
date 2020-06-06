import json

from easydict import EasyDict as edict


class Config:

    __init__ = lambda self: None

    def __new__(cls):
        if not hasattr(cls, '__config'):
            with open('config.json', 'r') as file:
                cls.__config = edict(json.load(file))
        return cls.__config

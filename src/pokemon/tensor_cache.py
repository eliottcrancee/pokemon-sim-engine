# tensor_cache.py

import torch


class OneHotCache:
    def __init__(self):
        self.cache = {}

    def get_one_hot(self, length, index):
        key = (length, index)
        if key not in self.cache:
            one_hot = torch.zeros(length)
            one_hot[index] = 1
            self.cache[key] = one_hot
        return self.cache[key]


def create_new_one_hot(length, index):
    one_hot = torch.zeros(length)
    one_hot[index] = 1
    return one_hot


ONEHOTCACHE = OneHotCache()

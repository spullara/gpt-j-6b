import time
import torch
from transformers import GPTNeoForCausalLM, AutoConfig, GPT2Tokenizer, PretrainedConfig
import torch
import transformers
from collections.abc import MutableMapping
import os

class Checkpoint(MutableMapping):
    def __init__(self):
        self.checkpoint = torch.load("pytorch_model.bin")
        print("Loaded")
    def __len__(self):
        return len(self.checkpoint)
    def __getitem__(self, key):
        return torch.load(self.checkpoint[key])
    def __setitem__(self, key, value):
        return
    def __delitem__(self, key):
        return
    def keys(self):
        return self.checkpoint.keys()
    def __iter__(self):
        for key in self.checkpoint:
            yield (key, self.__getitem__(key))
    def __copy__(self):
        return self.__dict__
    def copy(self):
        return self.__dict__

config = PretrainedConfig.from_json_file("config.json")

print("load", flush=True)
model = GPTNeoForCausalLM.from_pretrained(pretrained_model_name_or_path=None, config=config, state_dict=Checkpoint())
print("ok")
model.eval()
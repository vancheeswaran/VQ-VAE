import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
import numpy as np
import torch

class Data:
    
    def __init__(self):
        
        '''
        Initialisaing Data loader class
        '''
        
        self.training_data = datasets.CelebA(root=".", split="train", download=False, transform=transforms.Compose([ transforms.Resize([64,64]), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))]))
        
        self.validation_data = datasets.CelebA(root=".", split="valid", download=False, transform=transforms.Compose([ transforms.Resize([64,64]), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))]))
            
            
#         else:
            
#             self.training_data = datasets.KMNIST(root="data", train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))

#             self.validation_data = datasets.KMNIST(root="data", train=False, download=True, transform=transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.5,), (1.0,))]))
            
    def dataLoader(self):
        
        t,_=random_split(self.training_data, (55000, 107770))
        v,_=random_split(self.validation_data, (5500, 14367))
        
#         else:
            
#             t = self.training_data
#             v = self.validation_data
        
        return [t, v]
from model import Model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dataloader import Data
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class Test():
    
    '''
    Class to implement the algorithm with required parameters
    '''
    
    def __init__(self, batch_size = 256, num_updates = 15000, num_hiddens = 128, num_residual_hiddens = 32, num_residual_layers = 2, emb_dim = 64, num_embeddings = 512, commitment_cost = 0.25, learning_rate = 1e-3):
        
        '''
        Initialize the class variables with required parameter values
        '''
    
        self.batch_size = batch_size
        self.num_updates = num_updates
        self.num_hiddens = num_hiddens
        self.num_residual_hiddens = num_residual_hiddens
        self.num_residual_layers = num_residual_layers
        self.emb_dim = emb_dim 
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    def test(self):
        
        '''
        Runs the VQ-VAE algorithm with the parameters initialized
        
        :returns: 
        '''
        
        #initializing the Data class and calling the dataloader function
        dataloader = Data()
        train, valid = dataloader.dataLoader()
    
        #loading data for training and validation
        training_loader = DataLoader(train, 
                                     batch_size=self.batch_size, 
                                     shuffle=True,
                                     pin_memory=True)
        validation_loader = DataLoader(valid,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=True)
        
        #initializing the model and the optimizer
        model = Model(self.num_hiddens, self.num_residual_layers, self.num_residual_hiddens,
                      self.num_embeddings, self.emb_dim, 
                      self.commitment_cost).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, amsgrad=False)
        
        #training model
        model.train()
        train_recon_error = []
        train_perplexity = []

        for i in range(self.num_updates):
            (data, _) = next(iter(training_loader))
            data = data.to(self.device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = model(data)
            recon_error = F.mse_loss(data_recon, data)
            loss = recon_error + vq_loss

            loss.backward()

            optimizer.step()

            train_recon_error.append(recon_error.item())
            train_perplexity.append(perplexity.item())

            if (i+1) % 100 == 0:
                print('%d iterations' % (i+1))
                print('recon_error: %.3f' % np.mean(train_recon_error[-100:]))
                print('perplexity: %.3f' % np.mean(train_perplexity[-100:]))
                print()
                
        #reconstruction of the images as a grid
        model.eval()

        (original_imgs, _) = next(iter(validation_loader))
        original_imgs = original_imgs.to(self.device)

        prevq_img = model.pre_vq_conv(model.encoder(original_imgs))
        _, quantized_img, _, _ = model.vq_vae(prevq_img)
        reconstructed_imgs = model.decoder(quantized_img)
        
        return [original_imgs, reconstructed_imgs, train_recon_error, train_perplexity]
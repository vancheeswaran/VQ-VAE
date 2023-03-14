from vq_vae import VectorQuantizer
from decoder import Decoder
from encoder import Encoder
import torch.nn as nn

class Model(nn.Module):
    
    '''Class that extends existing nn.Module class to implement our model for VQ-VAE'''
    
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost):
        super(Model, self).__init__()
        
        '''
        Initializing the encoder, decoder, vqvae variables of the model
        '''
        
        self.encoder = Encoder(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self.decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x):
        
        '''
        Forward function of the VQ-VAE algorithm
        
        :param x: input image
        :type inputs: pytorch tensor
        :returns: loss, reconstructed x, perplexity
        '''
        
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        loss, quantized, perplexity, _ = self.vq_vae(z)
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity
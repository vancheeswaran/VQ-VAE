import torch
import torch.nn.functional as F
import torch.nn as nn


class ResidualBlock(nn.Module):
    
    '''
    This class implements the residual block.
    '''
    
    def __init__(self, input_channels, n_hidden, n_res_hidden):
        
        '''
        Initialising the residual block class
        '''
        
        super(ResidualBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels, out_channels=n_res_hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_res_hidden, out_channels=n_hidden, kernel_size=1, stride=1, bias=False)
            )
    
    def forward(self, x):
        
        '''
        Forward pass of the residual block
        '''
        
        return x + self.res_block(x)


class Residual(nn.Module):
    
    '''
    This class implements resnet
    '''
    
    def __init__(self, input_channels, n_hidden, n_res_layers, n_res_hidden):
        
        '''
        Initialising the residual class
        '''
        
        super(Residual, self).__init__()
        self.n_res_layers = n_res_layers
        self.layers = nn.ModuleList([ResidualBlock(input_channels, n_hidden, n_res_hidden) for i in range(self.n_res_layers)])

    def forward(self, x):
        
        '''
        Forward pass of the residual class
        '''
        
        for i in range(self.n_res_layers):
            x = self.layers[i](x)
        return F.relu(x)
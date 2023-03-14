import torch.nn.functional as F
import torch.nn as nn
from residual import Residual

class Decoder(nn.Module):
    
    '''
    This class implements the Decoder.
    '''
    
    def __init__(self, input_channels, n_hidden, n_res_layers, n_res_hidden):
        
        '''
        Initialising the Decoder class
        '''
        
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=n_hidden, kernel_size=3, stride=1, padding=1)
        
        self.residual = Residual(input_channels=n_hidden, n_hidden=n_hidden, n_res_layers=n_res_layers, n_res_hidden=n_res_hidden)
        
        self.trans_conv1 = nn.ConvTranspose2d(in_channels=n_hidden, out_channels=n_hidden//2, kernel_size=4, stride=2, padding=1)
        
        self.trans_conv2 = nn.ConvTranspose2d(in_channels=n_hidden//2, out_channels=3, kernel_size=4,stride=2, padding=1)

    def forward(self, inputs):
        
        '''
        Forward pass of the decoder layer
        '''
        
        x = self.conv1(inputs)
        
        x = self.residual(x)
        
        x = F.relu(self.trans_conv1(x))
        
        x = self.trans_conv2(x)
        
        return x
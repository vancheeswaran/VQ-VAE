import torch.nn.functional as F
import torch.nn as nn
from residual import Residual

class Encoder(nn.Module):
    
    '''
    This class implements the Encoder.
    '''
    
    def __init__(self, input_channels, n_hidden, n_res_layers, n_res_hidden):
        
        '''
        Initialising the Encoder class
        '''
        
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=n_hidden//2, kernel_size=4, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=n_hidden//2, out_channels=n_hidden, kernel_size=4, stride=2, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=n_hidden, out_channels=n_hidden, kernel_size=3, stride=1, padding=1)
        
        self.residual = Residual(input_channels=n_hidden, n_hidden=n_hidden, n_res_layers=n_res_layers, n_res_hidden=n_res_hidden)

    def forward(self, inputs):
        
        '''
        Forward pass of the encoder layer
        '''
        
        x = F.relu(self.conv1(inputs))
        
        x = F.relu(self.conv2(x))
        
        x = self.conv3(x)
        
        x = self.residual(x)
        
        return x
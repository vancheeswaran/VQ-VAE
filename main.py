import matplotlib.pyplot as plt
from test import Test
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class Experiment():
    def __init__(self, num_emb, com_cost, lr, updates):
        
        '''
        Initialize the variables for the experiment
        '''
        
        self.num_emb = num_emb
        self.com_cost = com_cost
        self.lr = lr
        self.updates = updates
        
    def show(self, img):
        
        '''
        Displays the images in the form of a grid
        '''
        
        npimg = img.numpy()
        fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        fig.axes.get_xaxis()
        fig.axes.get_yaxis()
    
    def run(self):
        
        '''
        Runs the entire model and plots error and perplexity graph and prints the original and reconstructed images
        '''
        
        sample = Test(batch_size = 256, num_updates = self.updates, num_hiddens = 128, num_residual_hiddens = 32, num_residual_layers = 2, emb_dim = 64, num_embeddings = self.num_emb, commitment_cost = self.com_cost, learning_rate = self.lr)

        original_imgs, reconstructed_imgs, train_recon_error, train_perplexity= sample.test()
        
        print("reconstructed images")
        self.show(make_grid(reconstructed_imgs.cpu().data)+0.4)

        print("original images", original_imgs.shape)
        self.show(make_grid(original_imgs.cpu())+0.4)
        

        f = plt.figure(figsize=(16,8))
        ax = f.add_subplot(1,2,1)
        ax.plot(train_recon_error)
        ax.set_yscale('log')
        ax.set_title('Smoothed NMSE.')
        ax.set_xlabel('iteration')

        ax = f.add_subplot(1,2,2)
        ax.plot(train_perplexity)
        ax.set_title('Smoothed Average codebook usage (perplexity).')
        ax.set_xlabel('iteration')


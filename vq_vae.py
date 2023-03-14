import torch
import torch.nn.functional as F
import torch.nn as nn

class VectorQuantizer(nn.Module):
    
    '''Extend the existing nn.Module class in torch to implement the vector quantizer'''
    
    
    def __init__(self, num_emb, emb_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        '''
        Intialize the variables required for the vector quantizer
        '''        
        
        self.emb_dim = emb_dim
        self.num_emb = num_emb
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_emb, self.emb_dim)
        self.embedding.weight.data.uniform_(-1/self.num_emb, 1/self.num_emb)
        

    def forward(self, inp):
        
        '''
        Forward function that performs the quantization
        
        :param in: input images
        :type inputs: pytorch tensor
        :returns: loss, quantized output, perplexity, encodings
        '''
    
        inp = inp.permute(0, 2, 3, 1).contiguous()
        
        flat_inp = inp.view(-1, self.emb_dim)
        
        d = torch.sum(flat_inp**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(flat_inp, self.embedding.weight.t())
            
        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encodings = (torch.zeros(encoding_indices.shape[0], self.num_emb, device=inp.device)).scatter_(1, encoding_indices, 1)
        
        e = torch.matmul(encodings, self.embedding.weight).view(inp.shape)
        
        loss = F.mse_loss(e, inp.detach()) + self.commitment_cost * F.mse_loss(e.detach(), inp)
        
        e = inp + (e - inp).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, e.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
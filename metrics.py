import torch
import torch.nn as nn

class Cos(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self,x,y):
        return self.cos(x,y) / self.temp

    def one_vs_all(self, x, i):
        return self(x[i:i+1],x)

class Euclidean(nn.Module):
    """
    Negative l2 distance
    """
    def __init__(self):
        super().__init__()

    def forward(self,x,y):
        return -((x-y)**2).mean(dim=-1)

    def one_vs_all(self, x, i):
        return self(x[i:i+1],x)

class ICM(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta=beta
        self.cos = nn.CosineSimilarity(dim=-1)
        self.precomputed=False

    def precompute(self,x):
        self.sq_mod=torch.sum(x**2,dim=1)
        self.mod=torch.sqrt(self.sq_mod)
        self.precomputed=True

    def forward(self,x ,y):
        sq_modx=torch.sum(x**2,dim=1)
        sq_mody=torch.sum(y**2,dim=1)
        term1=(1-self.beta)*(sq_modx+sq_mody)
        term2=self.beta*torch.sqrt(sq_modx)*torch.sqrt(sq_mody)*self.cos(x,y)
        return term1+term2     

    def one_vs_all(self, x, i):
        if not self.precomputed:
            self.precompute(x)
        term1=(1-self.beta)*(self.sq_mod[i]+self.sq_mod)
        term2=self.beta*self.mod[i]*self.mod*self.cos(x[i:i+1],x)
        return term1+term2

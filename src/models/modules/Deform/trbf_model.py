import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.rigid_utils import exp_se3
import os
from src.utils.system_utils import searchForMaxIteration
from src.utils.general_utils import get_expon_lr_func
from typing import Dict
import math

def trbfunction(x): 
    return torch.exp(-1*x.pow(2))

def create_sequence(n, L): # this gives you 0, 1, 2, ... L (L+1 entries)
    base_sequence = torch.arange(0, L+1).to(n.device)
    return (n ** base_sequence).view(-1, L+1, 1)
    #assert False, n.shape
    #assert False, (n**base_sequence).shape
    #return torch.cat((torch.tensor([1]).repeat(n.shape[0])[:, None].to(n.device), n ** base_sequence), dim=0).view(1, L+1, 1)

class TRBFModel(nn.Module):
    def __init__(self, 
            Np=3,
            Nq=1):
        self.Np = Np
        self.Nq = Nq
        super().__init__()
    def forward(self, inp: Dict, time: float):
        
        pointtimes = torch.ones((inp["means3D"].shape[0],1), dtype=inp["means3D"].dtype, requires_grad=False, device="cuda") + 0 # 
    
        trbfdistanceoffset = time * pointtimes -  inp["trbfcenter"]
        trbfdistance =  trbfdistanceoffset / torch.exp(inp["trbfscale"]) 
        trbfoutput = trbfunction(trbfdistance)
        
        opacities = inp["opacity"] * trbfoutput
        
        tforpoly = trbfdistanceoffset.detach()
        basis = create_sequence(tforpoly, self.Np) 
        # basis: Nx(Np+1)x1
        # inp["means3D"][:, :self.Np+1, :]: Nx(Np+1)x3
        means3D = torch.sum(inp["means3D"][:, :self.Np+1, :] * basis, dim=1) # Nx3
        #means3D = means3D[:, 0, :] +  measn3D[:, 1, :] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
        basis_rot = create_sequence(tforpoly, self.Nq)
        rotations = torch.sum(inp["rotations"][:, :self.Nq+1, :] * basis_rot, dim=1) #Nx4
        
        return means3D, rotations, 0., opacities, 0. 
        
        

    def train_setting(self, 
        **kwargs
        ):
        return None, {}
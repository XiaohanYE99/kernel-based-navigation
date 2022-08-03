import torch
import math
import numpy as np
from torch.autograd import Function
import time

class MultiCoverageLayer(Function):

    @staticmethod
    def forward(ctx, env, x):
        dL = torch.zeros(x.size(0),x.size(1)).to(env.device)
        poss=[]
        pos=x.detach().cpu().numpy().astype(float)
        for i in range(env.batch_size):
            poss.append(pos[i])

        loss=env.MCover.loss(poss)
        grad=env.MCover.grad()
        for i in range(env.batch_size):
            dL[i]=torch.from_numpy(grad[i]).squeeze().to(env.device)/env.batch_size
        ctx.save_for_backward(dL)

        return torch.Tensor(loss).to(env.device)

    @staticmethod
    def backward(ctx, grad_output):
        dL,=ctx.saved_tensors
        return None, dL


class CoverageLayer(Function):

    @staticmethod
    def forward(ctx, env, x):
        pos = x.detach().cpu().numpy().squeeze()
        #print(xx,pos)
        loss = env.Cover.loss(np.array(pos,dtype=float))
        return loss
    '''
    @staticmethod
    def backward(ctx, grad_output):
        dx, dv = ctx.saved_tensors

        return None, torch.matmul(grad_output, dx), torch.matmul(grad_output, dv)
        '''
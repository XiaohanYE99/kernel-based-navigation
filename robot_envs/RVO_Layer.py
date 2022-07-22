import torch
import math
import numpy as np
from torch.autograd import Function
import time
class MultiCollisionFreeLayer(Function):

    @staticmethod
    def forward(ctx, env, x, v, x_requires_grad=True):
        t0 = time.time()
        n=int(x.size(1)/2)
        xNew=np.empty(x.size())
        partial_x=torch.empty(x.size(0),x.size(1),x.size(1)).to(env.device)
        partial_v = torch.empty(x.size(0), x.size(1), x.size(1)).to(env.device)
        pos = x.detach().cpu().numpy()

        vx = v[:, :env.n_robots].detach().cpu().numpy()
        vy = v[:, env.n_robots:].detach().cpu().numpy()

        for i in range(env.n_robots):

            env.multisim.setAgentPrefVelocity(i, [(vx[j][i], vy[j][i]) for j in range(env.batch_size)])
            env.multisim.setAgentPosition(i, [(pos[j,i], pos[j,i+n]) for j in range(env.batch_size)])

        env.multisim.doNewtonStep(True,False,False)

        #env.sim.doStep()
        #pv=env.multisim.getGradV()
        #px=env.multisim.getGradX()

        for i in range(env.batch_size):
            partial_v[i] = torch.from_numpy(env.multisim.getGradV(i)).float()
            partial_x[i] = torch.from_numpy(env.multisim.getGradX(i)).float()
        #print(torch.sqrt(torch.sum(torch.square(partial_v))))
        for i in range(env.n_robots):
            pi=env.multisim.getAgentPosition(i)
            for b in range(env.batch_size):
                xNew[b,i ] = pi[b][0]
                xNew[b,i+n]=pi[b][1]

        #print(time.time() - t0)
        ctx.save_for_backward(partial_x, partial_v)
        return torch.from_numpy(xNew).float().to(env.device)

    @staticmethod
    def backward(ctx, grad_output):
        dx,dv=ctx.saved_tensors

        '''
        for i in range(dx.size(0)):
            if torch.max(torch.abs(dx[i]))>10:
                dx[i]=dx[i]*0
                dv[i]=dv[i]*0
        
        dx[dx>10]=0
        dx[dx<-10]=0
        dv[dv>10]=0
        dv[dv<-10]=0
        '''
        #return None, torch.matmul(grad_output.unsqueeze(1),dx).squeeze(1) , torch.matmul(grad_output.unsqueeze(1),dv).squeeze(1)
        return None, torch.matmul(dx,grad_output.unsqueeze(2)).squeeze(2), torch.matmul(dv,grad_output.unsqueeze(2)).squeeze(2)


class CollisionFreeLayer(Function):

    @staticmethod
    def forward(ctx, env, x, v, x_requires_grad=True):

        t0 = time.time()
        n = int(x.size(1) / 2)
        xNew = np.empty(x.size())
        partial_x = torch.empty(x.size(0), x.size(1), x.size(1)).to(env.device)
        partial_v = torch.empty(x.size(0), x.size(1), x.size(1)).to(env.device)
        pos = x.detach().cpu().numpy()
        vx = v[:, :env.n_robots].detach().cpu().numpy()
        vy = v[:, env.n_robots:].detach().cpu().numpy()
        for b in range(x.size(0)):
            for i in range(env.n_robots):
                dx = vx[b][i]
                dy = vy[b][i]

                env.sim.setAgentPrefVelocity(env.agent[i], (dx, dy))
                env.sim.setAgentPosition(env.agent[i], (pos[b][i], pos[b][i+n]))
                env.deltap[i] = [dx, dy]

            env.sim.doNewtonStep(True,False,False)
            # env.sim.doStep()
            p_v = env.sim.getGradV()
            p_x = env.sim.getGradX()

            partial_v[b] = torch.from_numpy(p_v).float()
            partial_x[b] = torch.from_numpy(p_x).float()

            for i in range(env.n_robots):
                xNew[b, i] = env.sim.getAgentPosition(env.agent[i])[0]
                xNew[b, i+n] = env.sim.getAgentPosition(env.agent[i])[1]
        #print(time.time()-t0)
        ctx.save_for_backward(partial_x, partial_v)
        return torch.from_numpy(xNew).float().to(env.device)

    @staticmethod
    def backward(ctx, grad_output):
        dx, dv = ctx.saved_tensors

        return None, torch.matmul(grad_output.unsqueeze(1),dx).squeeze(1) , torch.matmul(grad_output.unsqueeze(1),dv).squeeze(1)
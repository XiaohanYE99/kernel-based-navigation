import torch
import math
import numpy as np
from torch.autograd import Function
import time
class MultiCollisionFreeLayer(Function):

    @staticmethod
    def forward(ctx, env, x, v, x_requires_grad=True):
        t0 = time.time()
        x=x.reshape(-1,env.n_robots*2)
        xNew=np.empty(x.size())
        partial_x=torch.empty(x.size(0),x.size(1),x.size(1)).to(env.device)
        partial_v = torch.empty(x.size(0), x.size(1), x.size(1)).to(env.device)
        pos = x.reshape([-1, env.n_robots,2]).detach().cpu().numpy()
        vx = v[:, :env.n_robots].detach().cpu().numpy()
        vy = v[:, env.n_robots:].detach().cpu().numpy()

        for i in range(env.n_robots):

            env.multisim.setAgentPrefVelocity(i, [(vx[j][i], vy[j][i]) for j in range(env.batch_size)])
            env.multisim.setAgentPosition(i, [(pos[j,i,0], pos[j,i,1]) for j in range(env.batch_size)])

        env.multisim.doNewtonStep(True)

        #env.sim.doStep()
        pv=env.multisim.getGradV()
        px=env.multisim.getGradX()

        for i in range(env.batch_size):

            partial_v[i] = torch.from_numpy(pv[i]).float()
            partial_x[i] = torch.from_numpy(px[i]).float()
            print(torch.sum(partial_v[i]))
        for i in range(env.n_robots):
            pi=env.multisim.getAgentPosition(i)
            for b in range(env.batch_size):
                xNew[b,i * 2:i * 2 + 2] = pi[b]

        #print(time.time() - t0)
        ctx.save_for_backward(partial_x, partial_v)
        return torch.from_numpy(xNew).float().to(env.device),partial_v

    @staticmethod
    def backward(ctx, grad_output):
        dx,dv=ctx.saved_tensors
        return None, torch.matmul(grad_output,dx) , torch.matmul(grad_output,dv)


class CollisionFreeLayer(Function):

    @staticmethod
    def forward(ctx, env, x, v, x_requires_grad=True):

        t0 = time.time()
        x = x.reshape(-1, env.n_robots * 2)
        xNew = np.empty(x.size())
        partial_x = torch.empty(x.size(0), x.size(1), x.size(1)).to(env.device)
        partial_v = torch.empty(x.size(0), x.size(1), x.size(1)).to(env.device)
        pos = x.reshape([-1, env.n_robots, 2]).detach().cpu().numpy()
        vx = v[:, :env.n_robots].detach().cpu().numpy()
        vy = v[:, env.n_robots:].detach().cpu().numpy()
        for b in range(x.size(0)):
            for i in range(env.n_robots):
                dx = vx[b][i]
                dy = vy[b][i]

                env.sim.setAgentPrefVelocity(env.agent[i], (dx, dy))
                env.sim.setAgentPosition(env.agent[i], (pos[b][i][0], pos[b][i][1]))
                env.deltap[i] = [dx, dy]

            env.sim.doNewtonStep(True)
            # env.sim.doStep()
            p_v = env.sim.getGradV()
            p_x = env.sim.getGradX()
            partial_v[b] = torch.from_numpy(p_v).float()
            partial_x[b] = torch.from_numpy(p_x).float()
            for i in range(env.n_robots):
                xNew[b, i * 2:i * 2 + 2] = env.sim.getAgentPosition(env.agent[i])
        #print(time.time()-t0)
        ctx.save_for_backward(partial_x, partial_v)
        return torch.from_numpy(xNew).float().to(env.device),partial_v

    @staticmethod
    def backward(ctx, grad_output):
        dx, dv = ctx.saved_tensors
        return None, torch.matmul(grad_output, dx), torch.matmul(grad_output, dv)
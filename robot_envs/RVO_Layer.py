import torch
import math
import numpy as np
from torch.autograd import Function

class CollisionFreeLayer(Function):

    @staticmethod
    def forward(ctx, env, x, v, x_requires_grad=True):
        #t0 = time.time()
        x=x.reshape(-1,env.n_robots*2)
        xNew=np.empty(x.size())
        for b in range(x.size(0)):

            pos = x[b].reshape([-1, 2]).detach().cpu().numpy()
            vx = v[b,:env.n_robots]
            vy = v[b,env.n_robots:]
            vx = vx.detach().cpu().numpy()
            vy = vy.detach().cpu().numpy()

            for i in range(env.n_robots):
                dx = vx[i]
                dy = vy[i]
                # print(dx,dy)
                lenn = math.sqrt(dx * dx + dy * dy)
                '''
                if lenn > 2.0:

                    dx *= 2.0 / lenn
                    dy *= 2.0 / lenn
                    '''
                if env.state[i * 2] > 0.51 and env.state[i * 2] < 0.89 and env.state[i * 2 + 1] < 0.69 and \
                        env.state[i * 2 + 1] > 0.31:
                    r = np.sqrt((pow(pos[i][0] - 0.7, 2) + pow(pos[i][1] - 0.5, 2)))
                    dx = 1.0 * (0.7 - pos[i][0]) / r
                    dy = 1.0 * (0.5 - pos[i][1]) / r
                    if r < 0.01 :
                        dx = 0  # *=r/0.01
                        dy = 0  # *=r/0.01
                env.sim.setAgentPrefVelocity(env.agent[i], (dx, dy))
                env.sim.setAgentPosition(env.agent[i], (pos[i][0], pos[i][1]))
                env.deltap[i] = [dx, dy]

            env.sim.doNewtonStep(True)
            #env.sim.doStep()
            dv=torch.from_numpy(env.sim.getGradV()).float().to(env.device)
            dx=torch.from_numpy(env.sim.getGradX()).float().to(env.device)
            ctx.save_for_backward(dx,dv)
            for i in range(env.n_robots):
                xNew[b,i * 2:i * 2 + 2] = env.sim.getAgentPosition(env.agent[i])
            #print(time.time() - t0)
        return torch.from_numpy(xNew).float().to(env.device)

    @staticmethod
    def backward(ctx, grad_output):
        dx,dv=ctx.saved_tensors
        return None, torch.matmul(grad_output,dx) , torch.matmul(grad_output,dv)

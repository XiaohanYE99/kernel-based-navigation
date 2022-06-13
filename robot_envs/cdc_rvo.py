import torch
import math
from torch.autograd import Function

class CollisionFreeLayer(Function):
    def __init__(self, env):
        self.env=env
    @staticmethod
    def forward(self,ctx, x, v, x_requires_grad=True):
        velocity = v.squeeze(0)
        pos = x.reshape([-1, 2])
        vx = velocity[:self.env.n_robots]
        vy = velocity[self.env.n_robots:]
        vx = vx.detach().cpu().numpy()
        vy = vy.detach().cpu().numpy()
        for i in range(self.n_robots):
            dx = vx[i]
            dy = vy[i]
            # print(dx,dy)
            lenn = math.sqrt(dx * dx + dy * dy)
            if lenn > 2.0:
                dx *= 2.0 / lenn
                dy *= 2.0 / lenn

            if self.state[i * 2] > 0.51 and self.state[i * 2] < 0.89 and self.state[i * 2 + 1] < 0.69 and self.state[
                i * 2 + 1] > 0.31:
                r = np.sqrt((pow(pos[i][0] - 0.7, 2) + pow(pos[i][1] - 0.5, 2)))
                dx = 1.0 * (0.7 - pos[i][0]) / r
                dy = 1.0 * (0.5 - pos[i][1]) / r
                if r < 0.01:
                    dx = 0  # *=r/0.01
                    dy = 0  # *=r/0.01
            self.sim.setAgentPrefVelocity(self.agent[i], (dx, dy))
            self.sim.setAgentPosition(self.agent[i], (pos[i][0], pos[i][1]))
            self.deltap[i] = [dx, dy]


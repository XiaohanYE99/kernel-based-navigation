import torch
import numpy as np
import taichi as ti
import torch.nn as nn

import rvo2
from robot_envs.robot_env import *

class PolicyNet(nn.Module):
    def __init__(self, env, state_dim, action_dim, has_continuous_action_space):
        super(PolicyNet, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.env = env
        if has_continuous_action_space:
            self.action_dim = action_dim
        # actor
        if has_continuous_action_space:

            self.actor = nn.Sequential(
                nn.Linear(state_dim, 200),
                nn.Tanh(),
                nn.Linear(200, 200),
                nn.Tanh(),
                nn.Linear(200, 4 * self.env.N),
                nn.Sigmoid(),
            )
            '''
            self.actor = nn.Sequential(
                            nn.Conv2d(1, 8, 7, 2,3), nn.ReLU(),#[50,50,8]
                            nn.Conv2d(8, 12, 5, 2,2), nn.ReLU(),#[25,25,16]
                            nn.Conv2d(12, 20, 5, 2,2), nn.ReLU(), nn.Flatten(),#[13,13,32]
                            nn.Linear(20 * 13 * 13, 128), nn.ReLU(),
                            nn.Linear(128, action_dim),nn.Sigmoid()
                            )
            '''
        else:
            self.actor = nn.Sequential(
                nn.Conv2d(1, 8, 7, stride=2), nn.ReLU(),  # [50,50,8]
                nn.Conv2d(8, 12, 5, stride=2), nn.ReLU(),  # [25,25,16]
                nn.Conv2d(12, 20, 5, stride=2), nn.ReLU(), nn.Flatten(),  # [13,13,32]
                nn.Linear(20 * 13 * 13, 128), nn.ReLU(),
                nn.Linear(128, action_dim), nn.Sigmoid()
            )
        self.CFLayer = CollisionFreeLayer.apply

    def implement(self, state):
        action = torch.squeeze(self.actor(state), 1)
        # print(action)

        velocity = self.env.projection(action)

        v = self.env.get_velocity(state, velocity)

        xNew = self.CFLayer(self.env, state, v)

        return xNew


    def forward(self):
        raise NotImplementedError
def MBLoss(state,x,y):
    state=state.squeeze(0)
    half_size=int(state.size(0)/2)
    return -torch.sum(torch.square(state[:half_size]-x)+torch.square(state[half_size:]-y))
if __name__ == '__main__':
    iter = 1000
    steps = 300
    target_x=0.7
    target_y=0.5
    has_continuous_action_space = True  # continuous action space; else discrete
    device = torch.device('cpu')
    if (torch.cuda.is_available()):
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")

    use_kernel_loop = False  # calculating grid velocity with kernel loop
    use_sparse_FEM = False  # use sparse FEM solver

    gui = ti.GUI("DiffRVO", res=(500, 500), background_color=0x112F41)
    sim = rvo2.PyRVOSimulator(3 / 600., 0.03, 5, 0.04, 0.04, 0.01, 2)
    env = NavigationEnvs(gui, sim, use_kernel_loop, use_sparse_FEM)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = PolicyNet(env, state_dim, action_dim, has_continuous_action_space).to(device)

    opt = torch.optim.Adam(policy.actor.parameters(), lr=1e-4)
    for i in range(iter):
        state = env.reset()
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        for step in range(steps):
            state=policy.implement(state)
            env.MBStep(state)
        #loss=MBLoss(state,target_x,target_y)
        loss=env.MBLoss(state)
        loss.backward()
        opt.step()
        print('iter= ',i,'loss= ',loss)
        if i%3==0:
            torch.save(policy.actor,'model/model.pth')
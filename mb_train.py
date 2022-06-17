import torch
import numpy as np
import taichi as ti
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import random
from multiprocessing.dummy import Pool as ThreadPool

import rvo2
from robot_envs.robot_env import *

class PolicyNet(nn.Module):
    def __init__(self, env, state_dim, action_dim, has_continuous_action_space,action_std_init=0.4
                 ,horizon=200
                 ,num_sample_steps=5
                 ,num_pre_steps=10
                 ,num_train_steps=500
                 ,num_init_step=1
                 ,buffer_size=5000):
        super(PolicyNet, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        self.env = env
        self.buffer=[]
        self.buffer_size=buffer_size
        self.buffer_top=0
        self.isfull=False
        self.horizon=horizon
        self.num_sample_steps=num_sample_steps
        self.num_pre_steps=num_pre_steps
        self.num_init_step=num_init_step
        self.num_train_steps=num_train_steps

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
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
    def implement(self, state):
        I=self.env.P2G(state)
        I=I.unsqueeze(0).unsqueeze(0).to(device)
        action = torch.squeeze(self.actor(state), 1)
        # print(action)
        for i in range(self.env.N):
            self.env.x0[i]=action[0][4*i]
            self.env.y0[i]=action[0][4*i+1]
        velocity = self.env.projection(action)

        v = self.env.get_velocity(state.detach(), velocity)
        xNew = self.CFLayer(self.env, state.detach(), v)

        return xNew
    def act(self,state):
        I = self.env.P2G(state)
        I = I.unsqueeze(0).unsqueeze(0).to(device)
        action = torch.squeeze(self.actor(state), 1)

        for i in range(self.env.N):
            self.env.x0[i] = action[0][4 * i]
            self.env.y0[i] = action[0][4 * i + 1]
        velocity = self.env.projection(action)

        v = self.env.get_velocity(state.detach(), velocity)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(torch.zeros_like(v).to(device), cov_mat)
        v = v + dist.sample()

        xNew = self.CFLayer(self.env, state.detach(), v)

        return xNew
    def forward(self):
        raise NotImplementedError
    def train(self):
        loss_sum=0
        init_state=random.sample(self.buffer,self.num_train_steps)
        for state in init_state:

            state=state.to(device)
            state.requires_grad=True
            s=state
            for step in range(self.num_pre_steps):
                s = policy.implement(s)
                self.env.MBStep(s,render=False)
                #state=xNew

            loss = self.env.MBLoss(s)

            self.opt.zero_grad()
            loss.backward()
            #print(state.grad)
            self.opt.step()
            #print(loss.item())
            loss_sum+=loss.item()
        return loss_sum/self.num_train_steps
    def init_sample(self):
        for i in range(self.num_init_step):
            self.sample()
    def sample(self):
        for i in range(self.num_sample_steps):
            state = self.env.reset()
            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
            for step in range(self.horizon):
                with torch.no_grad():
                    state = policy.act(state)
                    self.env.MBStep(state)

                if not self.isfull:
                    self.buffer.append(state.cpu())
                else:
                    self.buffer[self.buffer_top]=state.cpu()

                self.buffer_top+=1
                if self.buffer_top==self.buffer_size:
                    self.is_full=True
                    self.buffer_top=0

if __name__ == '__main__':
    iter = 300
    steps = 50
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
    sim = rvo2.PyRVOSimulator(1 / 200., 0.03, 5, 0.04, 0.04, 0.01, 2)
    env = NavigationEnvs(gui, sim, use_kernel_loop, use_sparse_FEM)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = PolicyNet(env, state_dim, action_dim, has_continuous_action_space).to(device)
    #policy.actor=torch.load('model/model_30.pth')
    #init sample
    policy.init_sample()
    for i in range(iter):
        policy.sample()
        loss=policy.train()
        print('iter= ',i,'loss= ',loss)
        if i%10==0:
            torch.save(policy.actor,'model/model_%d.pth'%i)

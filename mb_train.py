import torch
import numpy as np
import taichi as ti
import torch.nn as nn
from torch.autograd import Function
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import random


import rvo2,pyglet
from robot_envs.robot_env import *
from robot_envs.RVO_Layer import CollisionFreeLayer,MultiCollisionFreeLayer


class PolicyNet(nn.Module):
    def __init__(self, env, state_dim, action_dim, has_continuous_action_space, action_std_init=0.1
                 , horizon=150
                 , num_sample_steps=12
                 , num_pre_steps=10
                 , num_train_steps=64* 25
                 , num_init_step=0
                 , buffer_size=1800
                 , batch_size=64):
        super(PolicyNet, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        self.env = env
        self.buffer = []
        self.target_buffer = []
        self.buffer_size = buffer_size
        self.buffer_top = 0
        self.isfull = False
        self.horizon = horizon
        self.num_sample_steps = num_sample_steps
        self.num_pre_steps = num_pre_steps
        self.num_init_step = num_init_step
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        if has_continuous_action_space:
            self.action_dim = action_dim
        # actor
        if has_continuous_action_space:

            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 5 * self.env.N),
                nn.Sigmoid(),
            )
            '''
            self.actor = nn.Sequential(
                nn.Conv2d(1, 8, 7, 2, 3),  nn.BatchNorm2d(8),nn.ReLU(),  # [50,50,8]
                nn.Conv2d(8, 12, 5, 2, 2),nn.BatchNorm2d(12), nn.ReLU(),  # [25,25,16]
                nn.Conv2d(12, 20, 5, 2, 2), nn.BatchNorm2d(20),nn.ReLU(), nn.Flatten(),  # [13,13,32]
                nn.Linear(20 * 13 * 13, 128),nn.ReLU(),
                nn.Linear(128, action_dim), nn.Sigmoid()
            )
            '''
            for name, param in self.actor.named_parameters():
                if (len(param.size()) >= 2):
                    nn.init.kaiming_uniform_(param, a=1e-3)
            self.lr = 3e-4
            self.opt = torch.optim.Adam([{'params': self.actor.parameters(), 'lr': self.lr}])

        self.CFLayer = CollisionFreeLayer.apply
        self.MultiCFLayer = MultiCollisionFreeLayer.apply

    def switch(self, v,state,target):
        x=state[:,::2]-target[0]
        y=state[:,1::2]-target[1]
        r=torch.sqrt(x*x+y*y)+self.env.eps
        alpha=-3*torch.pow((self.env.bound-r),2)/(r*r)+2*torch.pow((self.env.bound-r),3)/(r*r*r)+1.0
        alpha=F.relu(alpha)
        alpha=1.0-F.relu(1.0-alpha)
        alpha=torch.cat((alpha,alpha),1)
        #print(alpha)
        return (1.0-alpha)*v+alpha*(-torch.cat((x/r,y/r),1))


    def implement(self, state, target,training):
        I = self.env.P2G(state, target)
        I = I.to(device)

        # action = self.controller(I)

        action = torch.squeeze(self.actor(state), 1)

        for i in range(self.env.N):
            self.env.x0[i] = action[0][5 * i]
            self.env.y0[i] = action[0][5 * i + 1]

        velocity = self.env.projection(action)

        v = self.env.get_velocity(state, velocity)
        v = self.switch(v, state, target)
        if training:
            xNew = self.MultiCFLayer(self.env, state, v)
        else:
            xNew = self.CFLayer(self.env, state, v)

        return xNew

    def act(self, state):
        I = self.env.P2G(state, [0])
        I = I.to(device)
        action = torch.squeeze(self.actor(state), 1)

        for i in range(self.env.N):
            self.env.x0[i] = action[0][5 * i]
            self.env.y0[i] = action[0][5 * i + 1]
        velocity = self.env.projection(action)

        v = self.env.get_velocity(state.detach(), velocity)

        xNew = self.CFLayer(self.env, state.detach(), v)

        return xNew

    def forward(self):
        raise NotImplementedError

    '''
    def train(self):
        loss_sum=0
        init_state=random.sample(self.buffer,self.num_train_steps)
        for state in init_state:
            loss=0
            state=state.to(device)
            state.requires_grad=True
            #s=state
            for step in range(self.num_pre_steps):
                state = policy.implement(state)
                #self.env.MBStep(state,render=False)
                loss += self.env.MBLoss(state)
            self.opt.zero_grad()
            loss.backward()
            print(state.grad)
            self.opt.step()
            #print(loss.item())
            loss_sum+=loss.item()
        return loss_sum/self.num_train_steps
        '''

    def reset_env(self):
        self.env.cnt = 0
        self.env.target = random.randint(1, 4)
        if self.env.target == 1:
            self.env.aim = [0.1, 0.5]
        elif self.env.target == 2:
            self.env.aim = [0.9, 0.5]
        elif self.env.target == 3:
            self.env.aim = [0.5, 0.1]
        elif self.env.target == 4:
            self.env.aim = [0.5, 0.9]
        # self.env.aim = [0.5, 0.9]
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                self.env.state[idx * 2:idx * 2 + 2] = [0.46 + i * 0.02, 0.46 + j * 0.02]
                self.env.sim.setAgentPosition(self.env.agent[idx],
                                              (self.env.state[idx * 2], self.env.state[idx * 2 + 1]))

        return self.env.state

    def update(self):

        loss_sum = 0
        init_state = random.sample(self.buffer, self.num_train_steps)
        init_target = random.sample(self.target_buffer, self.num_train_steps)
        t = int(self.num_train_steps / self.batch_size)
        for i in range(t):
            state_batch = np.array(init_state[i * self.batch_size:(i + 1) * self.batch_size],
                                   dtype=np.float32).squeeze()
            target = np.array(init_target[i * self.batch_size:(i + 1) * self.batch_size],
                              dtype=np.float32).squeeze()
            state = torch.from_numpy(state_batch).to(device)
            state.requires_grad = True
            s = state
            loss = 0

            for step in range(self.num_pre_steps):
                s = policy.implement(s, self.env.aim,training=True)

                # s=xNew
            loss = self.env.MBLoss(s, state)
            self.opt.zero_grad()
            loss.backward()
            #print(state.grad)
            self.opt.step()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 2)
            # print(loss.item())
            loss_sum += loss
        return loss_sum / self.num_train_steps

    def init_sample(self):
        for i in range(self.num_init_step):
            self.sample(use_random_policy=True)

    def sample(self, use_random_policy=False):

        for i in range(self.num_sample_steps):
            state = self.env.reset()
            # state=self.reset_env()
            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
            for step in range(self.horizon):
                with torch.no_grad():
                    if use_random_policy:
                        state = policy.implement(state, self.env.aim,training=False)
                    else:
                        state = policy.implement(state, self.env.aim,training=False)
                    self.env.MBStep(state)

                if not self.isfull:
                    self.buffer.append(state.tolist())
                    self.target_buffer.append(self.env.target)
                else:
                    self.buffer[self.buffer_top] = state.tolist()
                    self.target_buffer[self.buffer_top] = self.env.target

                self.buffer_top += 1
                if self.buffer_top == self.buffer_size:
                    self.is_full = True
                    self.buffer_top = 0

    def test(self):
        for i in range(self.num_sample_steps):
            state = self.env.reset()
            # state = self.reset_env()
            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
            for step in range(self.horizon):
                with torch.no_grad():
                    state = policy.implement(state)
                    self.env.MBStep(state)


if __name__ == '__main__':
    iter = 100
    steps = 100
    target_x = 0.7
    target_y = 0.5
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

    batch_size = 64
    gui = ti.GUI("DiffRVO", res=(500, 500), background_color=0x112F41)
    sim = rvo2.PyRVOSimulator(3 / 400., 0.03, 5, 0.04, 0.04, 0.01, 2)
    multisim = rvo2.PyRVOMultiSimulator(batch_size,3 / 400., 0.03, 5, 0.04, 0.04, 0.01, 2)

    env = NavigationEnvs(batch_size, gui, sim, multisim, use_kernel_loop, use_sparse_FEM,fn="./robot_envs/mazes_g100w700h700_Var10/maze18.dat")
    def step(dt):
        env.render()
    pyglet.clock.schedule_interval(step,1.0/60)
    pyglet.app.run()
    '''
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = PolicyNet(env, state_dim, action_dim, has_continuous_action_space,batch_size=batch_size).to(device)
    #policy.actor = torch.load('model/model_12.pth')
    # init sample
    # policy.eval()
    policy.init_sample()
    for i in range(iter):
        if i in [30, 70]:
            policy.lr *= 0.3
        #policy.eval()
        policy.sample(False)

        # policy.test()
        policy.train()
        loss = 0
        for k in range(1):
            loss += policy.update()
        print('iter= ', i, 'loss= ', loss)
        if i % 1 == 0:
            torch.save(policy.actor, 'model/model_%d.pth' % i)
    '''
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
    def __init__(self, env, state_dim, action_dim, has_continuous_action_space, action_std_init=0.8
                 , horizon=64
                 , num_sample_steps=1
                 , num_pre_steps=1
                 , num_train_steps=64
                 , num_init_step=0
                 , buffer_size=64
                 , batch_size=64):
        super(PolicyNet, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_var = torch.full((100,), action_std_init * action_std_init).to(device)
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
            '''
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(64, 128),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(128, 5 * self.env.N),

                #nn.Sigmoid(),
            )
            '''
            self.actor = nn.Sequential(
                nn.Conv2d(3, 8, 7, 2, 3), nn.BatchNorm2d(8), nn.ReLU(),  # [50,50,8]
                nn.Conv2d(8, 12, 5, 2, 2), nn.BatchNorm2d(12), nn.ReLU(),  # [25,25,16]
                nn.Conv2d(12, 20, 5, 2, 2),nn.BatchNorm2d(20), nn.ReLU(), nn.Flatten(),  # [9,9,128]
                nn.Linear(20 * 7 * 7, 128),nn.ReLU(),
                nn.Linear(128, action_dim), nn.Sigmoid()
            )


            for name, param in self.actor.named_parameters():
                if (len(param.size()) >= 2):
                    nn.init.kaiming_uniform_(param, a=1e-3)

            self.lr = 1e-4
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
        return (1.0-alpha)*v+alpha*(-3*torch.cat((x/r,y/r),1))


    def implement(self, state, target,training):
        I = self.env.P2G(state, target)
        I = I.to(device)
        # action = self.controller(I)

        action = torch.squeeze(self.actor(I), 1)#+0.5

        for i in range(self.env.N):
            self.env.x0[i] = action[0][5 * i]
            self.env.y0[i] = action[0][5 * i + 1]

        velocity = self.env.projection(action)

        v = self.env.get_velocity(state/self.env.scale, velocity)

        v = self.switch(v, state/self.env.scale, target)

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

    def update(self):

        loss_sum = 0
        init_state =self.buffer# random.sample(self.buffer, self.num_train_steps)

        init_target = random.sample(self.target_buffer, self.num_train_steps)
        t = int(self.num_train_steps / self.batch_size)
        for i in range(t):

            state_batch = np.array(init_state[i * self.batch_size:(i + 1) * self.batch_size],
                                   dtype=np.float32).squeeze()
            target = np.array(init_target[i * self.batch_size:(i + 1) * self.batch_size],
                              dtype=np.float32).squeeze()
            state = torch.from_numpy(state_batch).to(device)
            #print(torch.sum(state))
            state.requires_grad = True
            s = state
            loss = 0

            for step in range(self.num_pre_steps):
                xNew = policy.implement(s, self.env.aim,training=True)
                loss += self.env.MBLoss(xNew, s)
                s=xNew

            self.opt.zero_grad()
            #with torch.autograd.detect_anomaly():
            loss.backward()
            #print(state.grad)
            self.opt.step()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
            # print(loss.item())
            loss_sum += loss
        return loss_sum / self.num_train_steps

    def init_sample(self):
        for i in range(self.num_init_step):
            self.sample(use_random_policy=True)

    def sample(self, use_random_policy=False):
        loss=0
        self.buffer =[]
        for i in range(self.num_sample_steps):
            state = self.env.reset_agent()
            # state=self.reset_env()
            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
            s=state
            for step in range(self.horizon):
                with torch.no_grad():
                    #print(state, state.tolist())
                    if use_random_policy:
                        state = policy.implement(state, self.env.aim,training=False)
                    else:
                        state = policy.implement(state, self.env.aim,training=False)
                    self.env.MBStep(state)

                self.buffer.append(state.tolist())
                self.target_buffer.append(self.env.target)

            loss+=self.env.MBLoss(state,s)
        return loss
    def test(self):
        for i in range(self.num_sample_steps):
            state = self.env.reset_agent()
            # state = self.reset_env()
            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
            for step in range(self.horizon):
                with torch.no_grad():
                    state = policy.implement(state)
                    self.env.MBStep(state)
    def reset(self,path='./robot_envs/mazes_g100w700h700/maze'):
        idx=random.randint(0,400)
        fn=path+str(idx)+'.dat'
        self.env.load_roadmap(fn)

if __name__ == '__main__':
    iter = 3000
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
    sim = rvo2.PyRVOSimulator(800 / 400., 10, 100, 1.5, 2.0, 8, 2)
    multisim = rvo2.PyRVOMultiSimulator(batch_size,800 / 400., 10, 100, 1.5, 2.0, 8, 2)

    env = NavigationEnvs(batch_size, gui, sim, multisim, use_kernel_loop, use_sparse_FEM)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = PolicyNet(env, state_dim, action_dim, has_continuous_action_space,batch_size=batch_size).to(device)
    #policy.actor = torch.load('model/model_450.pth')
    # init sample
    # policy.eval()
    #policy.init_sample()
    sumloss=0
    for i in range(iter):
        if i in [300, 1000]:
            policy.lr *= 0.3

        policy.reset()
        #policy.eval()
        loss=policy.sample(False)

        # policy.test()
        policy.train()
        for k in range(1):
            policy.update()
        torch.cuda.empty_cache()
        print('iter= ', i, 'loss= ', loss)
        sumloss+=loss
        if i % 10 == 0:
            torch.save(policy.actor, 'model/model_%d.pth' % i)
        if i%50==0:
            print('iter=',i,'sumloss= ',sumloss/50)
            sumloss=0
    pyglet.app.run()
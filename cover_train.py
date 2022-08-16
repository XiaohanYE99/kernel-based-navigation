import torch
import numpy as np
import taichi as ti
import torch.nn as nn
from torch.autograd import Function
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import random


import pyglet
import pyRVO as pyrvo
from robot_envs.cover_env import *
from robot_envs.RVO_Layer import CollisionFreeLayer,MultiCollisionFreeLayer
from robot_envs.Coverage_Layer import CoverageLayer,MultiCoverageLayer


class PolicyNet(nn.Module):
    def __init__(self, env, state_dim, action_dim, has_continuous_action_space, action_std_init=0.2
                 , horizon=256
                 , num_sample_steps=1
                 , num_pre_steps=1
                 , num_train_steps=256
                 , num_init_step=0
                 , buffer_size=256
                 , batch_size=128):
        super(PolicyNet, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space

        self.env = env
        self.action_var = torch.full((75,), action_std_init * action_std_init).to(device)
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
                nn.Conv2d(1, 8, 5, 1, 0,bias=False), nn.MaxPool2d(2),  nn.Tanh(),  # [50,50,8]
                nn.Conv2d(8, 12, 3, 1, 0,bias=False), nn.MaxPool2d(2),  nn.Tanh(),  # [25,25,16]
                nn.Conv2d(12, 20, 3, 1, 0,bias=False), nn.MaxPool2d(2),  nn.Tanh(), nn.Flatten(),  # [9,9,128]
                nn.Linear(20 * 10 * 10, 128,bias=False), nn.Dropout(0.25), nn.Tanh(),
                nn.Linear(128, action_dim,bias=False)#, nn.Sigmoid()
            )

            for name, param in self.actor.named_parameters():
                if (len(param.size()) >= 2):
                    nn.init.kaiming_uniform_(param, a=1.5e-0)
            self.actor = torch.load('model/model_7.pth')
            self.lr = 1e-4*0.3*0.1
            self.opt = torch.optim.Adam([{'params': self.actor.parameters(), 'lr': self.lr,'weight decay':0}])


        self.CFLayer = CollisionFreeLayer.apply
        self.MultiCFLayer = MultiCollisionFreeLayer.apply
        self.CoverageLayer=CoverageLayer.apply
        self.MultiCoverageLayer=MultiCoverageLayer.apply

    def implement(self, state, target,training):
        I = self.env.P2G(state)
        I = I.to(device)
        # action = self.controller(I)

        action = torch.squeeze(self.actor(I), 1)+0.5

        for i in range(self.env.N):
            self.env.x0[i] = action[0][5 * i]
            self.env.y0[i] = action[0][5 * i + 1]

        velocity = self.env.projection(action)

        v = self.env.get_velocity(state/self.env.scale, velocity,is_norm=False)

        state_f=torch.zeros_like(state)
        v_f=torch.zeros_like(v)
        state_f[:,::2]=state[:,:self.env.n_robots]
        state_f[:, 1::2] = state[:, self.env.n_robots:]
        v_f[:, ::2] = v[:, :self.env.n_robots]
        v_f[:, 1::2] = v[:, self.env.n_robots:]
        if training:
            xNew_f = self.MultiCFLayer(self.env, state_f, v_f)
        else:
            xNew_f = self.CFLayer(self.env, state_f, v_f)
        xNew=torch.cat((xNew_f[:,::2],xNew_f[:,1::2]),1)
        return xNew

    def update(self):

        loss_sum = 0
        init_state =random.sample(self.buffer, self.num_train_steps)

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
                #loss += self.env.MBLoss(xNew, s)
                s=xNew
            loss = self.env.ShapeLoss(s,state)
            self.opt.zero_grad()
            #with torch.autograd.detect_anomaly():

            loss.backward()

            print(torch.max(torch.abs(state.grad)))
            nn.utils.clip_grad_value_(self.actor.parameters(), 2)
            self.opt.step()
            # print(loss.item())
            loss_sum += loss.data#.item()
        #print(self.actor.state_dict()['0.weight'])
        #print(self.env.aim)
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
                    #print(self.CoverageLayer(self.env,sf))
                self.buffer.append(state.tolist())
                self.target_buffer.append(self.env.target)
                #print(state)
            loss+=self.env.ShapeLoss(state,s)
        return loss

    def reset(self,path='./robot_envs/mazes_g75w675h675/maze'):
        idx=random.randint(0,500)
        idx=78
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


    sim = pyrvo.RVOSimulator(2,84,1e-4,1,0.5,200)
    multisim = pyrvo.MultiRVOSimulator(batch_size,2,84,1e-4,1,0.5,200)

    env = CoverEnvs(batch_size, gui, sim, multisim, use_kernel_loop, use_sparse_FEM)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = PolicyNet(env, state_dim, action_dim, has_continuous_action_space,batch_size=batch_size).to(device)
    #policy.actor = torch.load('model/model_10.pth')
    # init sample
    # policy.eval()
    #policy.init_sample()
    sumloss = 0
    trainlosssum = 0

    policy.reset()
    for i in range(iter):
        if i in [50]:
            policy.lr *= 0.3

        #policy.env.reset()
        policy.eval()
        loss = policy.sample(False)

        # policy.test()
        trainloss = 0

        policy.train()
        for k in range(0):
            trainloss += policy.update()
        trainlosssum += trainloss
        torch.cuda.empty_cache()
        print('iter= ', i, 'loss= ', loss, 'trainloss= ', trainloss)
        sumloss += loss
        #if i % 10 == 0:
        #if loss.data<1e-4:
        #    torch.save(policy.actor, 'model/model_%d.pth' % i)
        if i % 40 == 0:
            print('iter=', i, 'sumloss= ', sumloss / 40, 'suntrainloss= ', trainlosssum / 40)
            sumloss = 0
            trainlosssum = 0
    pyglet.app.run()
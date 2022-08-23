import torch
import torch.nn as nn
import random
import time
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import time
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.targets=[]
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.targets[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, env,state_dim, action_dim, has_continuous_action_space, action_std_init,CollisionFreeLayer):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.env=env
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            '''
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 200),
                            nn.Tanh(),
                            nn.Linear(200, 200),
                            nn.Tanh(),
                            nn.Linear(200, 5*self.env.N),
                            nn.Sigmoid(),
                        )
            '''
            self.actor = nn.Sequential(
                nn.Conv2d(2, 8, 5, 1, 0, bias=False), nn.MaxPool2d(2), nn.Tanh(),  # [50,50,8]
                nn.Conv2d(8, 12, 5, 1, 0, bias=False), nn.MaxPool2d(2), nn.Tanh(),  # [25,25,16]
                nn.Conv2d(12, 20, 3, 1, 0, bias=False), nn.MaxPool2d(2), nn.Tanh(),  # [25,25,16]
                nn.Conv2d(20, 32, 3, 1, 0, bias=False), nn.Tanh(), nn.Flatten(),  # [9,9,128]
                nn.Linear(32 * 8 * 8, 128, bias=False), nn.Tanh(), nn.Dropout(0.25),
                nn.Linear(128, action_dim, bias=False), nn.Sigmoid()
            )
        # critic

        self.critic = nn.Sequential(
                nn.Conv2d(2, 8, 5, 1, 0,bias=False), nn.MaxPool2d(2),  nn.Tanh(),  # [50,50,8]
                nn.Conv2d(8, 12, 5, 1, 0,bias=False), nn.MaxPool2d(2),  nn.Tanh(),  # [25,25,16]
                nn.Conv2d(12, 20, 3, 1, 0,bias=False), nn.MaxPool2d(2),  nn.Tanh(),  # [25,25,16]
                nn.Conv2d(20, 32, 3, 1, 0,bias=False),   nn.Tanh(), nn.Flatten(),  # [9,9,128]
                nn.Linear(32 * 8 * 8, 128,bias=False),  nn.Tanh(),nn.Dropout(0.25),
                nn.Linear(128, 1,bias=False)
            )

        self.CFLayer = CollisionFreeLayer.apply
    def switch(self, v,state,target):
        x = state[:, :self.env.n_robots] - target[0]
        y = state[:, self.env.n_robots:] - target[1]

        r = torch.sqrt(torch.square(x) + torch.square(y) + 1e-2)
        alpha = -3 * torch.pow((2 * self.env.bound - r), 2) / (r * r) + 2 * torch.pow((2 * self.env.bound - r), 3) / (
                    r * r * r) + 1.0

        alpha[r < self.env.bound] = 0
        alpha[r > 2 * self.env.bound] = 1
        alpha = torch.cat((alpha, alpha), 1)
        return alpha*v+(1.0-alpha)*(-5*torch.cat((x/r,y/r),1))
    def implement(self,I):
        #print(state)
        # action = self.controller(I)

        action = torch.squeeze(self.actor(I), 1)

        return action
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state,target):
        I = self.env.P2G(state)
        I = I.to(device)
        if self.has_continuous_action_space:
            action_mean = self.implement(I)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.implement(I)
            dist = Categorical(action_probs)

        action = dist.sample()
        
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action,target):
        t0 = time.time()

        I = self.env.P2G(state, target)
        I = I.to(device)
        if self.has_continuous_action_space:
            
            action_mean = self.implement(I)

            #print(action_mean.size())
            action_var = self.action_var.expand_as(action_mean)
            
            cov_mat = torch.diag_embed(action_var).to(device)


            dist = MultivariateNormal(action_mean, cov_mat)


            #print(torch.cuda.memory_reserved())
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.implement(I)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_values = self.critic(I)

        #print(time.time() - t0)
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self,env, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,CollisionFreeLayer, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(env,state_dim, action_dim, has_continuous_action_space, action_std_init,CollisionFreeLayer).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        
        self.policy_old = ActorCritic(env,state_dim, action_dim, has_continuous_action_space, action_std_init,CollisionFreeLayer).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                #print(state)

                action, action_logprob = self.policy_old.act(state,self.policy.env.aim)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.targets.append(self.policy.env.target_map)
            
            return action#.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.targets.append(self.policy.env.target_map)
            return action.item()

    def update(self):
        # Monte Carlo estimate of returns

        batch=32
        rewards_ = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards_ = torch.tensor(rewards_, dtype=torch.float32).to(device)
        rewards_ = (rewards_ - rewards_.mean()) / (rewards_.std() + 1e-7)
        
        # convert list to tensor
        old_states_ = torch.squeeze(torch.stack(self.buffer.states, dim=0),1).detach().to(device)
        old_actions_ = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs_ = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        targets_=self.buffer.targets
        nums=old_states_.size(0)
        # Optimize policy for K epochs
        #idx=[i for i in range(nums)]
        for _ in range(self.K_epochs):
            K=int(nums/batch)
            for i in range(K):
                t0=time.time()

                #a=random.sample(idx, batch)
                old_states=old_states_[i::K]#old_states_[i*batch:i*batch+batch]
                old_actions=old_actions_[i::K]
                old_logprobs=old_logprobs_[i::K]
                targets=targets_[i::K]
                rewards=rewards_[i::K]
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions,targets)
                
                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())
    
                # Finding Surrogate Loss
                advantages = rewards - state_values.detach()   
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
    
                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()

                self.optimizer.step()
                #print(time.time()-t0)
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        #print(time.time()-t0)
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
    def step(self, state,action,target):
        #state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        velocity = self.policy.env.projection(action)

        v = self.policy.env.get_velocity(state / self.policy.env.scale, velocity)
        v = self.policy.switch(v, state / self.policy.env.scale, target)
        state_f = torch.zeros_like(state)
        v_f = torch.zeros_like(v)
        state_f[:, ::2] = state[:, :self.policy.env.n_robots]
        state_f[:, 1::2] = state[:, self.policy.env.n_robots:]
        v_f[:, ::2] = v[:, :self.policy.env.n_robots]
        v_f[:, 1::2] = v[:, self.policy.env.n_robots:]

        xNew_f = self.policy.CFLayer(self.policy.env, state_f, v_f)

        xNew = torch.cat((xNew_f[:, ::2], xNew_f[:, 1::2]), 1)
        done=0
        reward=-10*self.policy.env.MBLoss(state,xNew).item()
        self.policy.env.MBStep(state)
        return xNew, reward, done, dict(reward=reward)
    def reset(self,path='./robot_envs/mazes_g75w675h675/maze'):
        idx=random.randint(0,500)
        idx=78
        fn=path+str(idx)+'.dat'
        self.policy.env.load_roadmap(fn)
    def sample(self,maxlen):
        loss = 0
        for i in range(4):
            self.policy.env.reset()
            state = self.policy.env.reset_agent()
            # state=self.reset_env()
            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
            s = state
            for step in range(maxlen):

                with torch.no_grad():
                    I = self.policy.env.P2G(state, self.policy.env.aim)
                    I = I.to(device)
                    action = self.policy.implement(I)
                    for i in range(self.policy.env.N):
                        self.policy.env.x0[i] = action[0][5 * i]
                        self.policy.env.y0[i] = action[0][5 * i + 1]
                    state,_,_,_=self.step(state,action,self.policy.env.aim)
                    self.policy.env.MBStep(state)
            loss += self.policy.env.MBLoss(state, s)

        return loss/4



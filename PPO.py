import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from robot_envs.robot_env import CollisionFreeLayer
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
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, env,state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.env=env
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 200),
                            nn.Tanh(),
                            nn.Linear(200, 200),
                            nn.Tanh(),
                            nn.Linear(200, 4*self.env.N),
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
                            nn.Conv2d(1, 8, 7, stride=2), nn.ReLU(),#[50,50,8]
                            nn.Conv2d(8, 12, 5, stride=2), nn.ReLU(),#[25,25,16]
                            nn.Conv2d(12, 20, 5, stride=2), nn.ReLU(), nn.Flatten(),#[13,13,32]
                            nn.Linear(20 * 13 * 13, 128), nn.ReLU(),
                            nn.Linear(128, action_dim),nn.Sigmoid()
                        )
        # critic
        
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 200),
                        nn.Tanh(),
                        nn.Linear(200, 200),
                        nn.Tanh(),
                        nn.Linear(200, 1)
                    )
        '''
        self.critic = nn.Sequential(
                            nn.Conv2d(1, 8, 7, 2,3), nn.ReLU(),#[50,50,8]
                            nn.Conv2d(8, 12, 5, 2,2), nn.ReLU(),#[25,25,16]
                            nn.Conv2d(12, 20, 5, 2,2), nn.ReLU(), nn.Flatten(),#[13,13,32]
                            nn.Linear(20 * 13 * 13, 128), nn.ReLU(),
                            nn.Linear(128, 1)
                            )
        '''
        self.CFLayer = CollisionFreeLayer.apply
    def implement(self,state):

        action=torch.squeeze(self.actor(state),1)
        #print(action)

        velocity=self.env.projection(action)

        v=self.env.get_velocity(state,velocity)

        xNew=self.CFLayer(self.env,state,v)

        return xNew
    '''
    def implement(self,img):
        output=torch.squeeze(self.actor(img),1)

        velocity=self.env.projection1(output,img)
        return velocity
    '''
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.implement(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.implement(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        
        if self.has_continuous_action_space:
            
            action_mean = self.implement(state)  

            #print(action_mean.size())
            action_var = self.action_var.expand_as(action_mean)
            
            cov_mat = torch.diag_embed(action_var).to(device)


            dist = MultivariateNormal(action_mean, cov_mat)


            #print(torch.cuda.memory_reserved())
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.implement(state)
            dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self,env, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(env,state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        
        self.policy_old = ActorCritic(env,state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
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
                state = torch.unsqueeze(torch.FloatTensor(state),0).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            
            return action#.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        t0=time.time()
        batch=100
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
        nums=old_states_.size(0)
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for i in range(int(nums/batch)):
                
                old_states=old_states_[i*batch:i*batch+batch]
                old_actions=old_actions_[i*batch:i*batch+batch]
                old_logprobs=old_logprobs_[i*batch:i*batch+batch]
                rewards=rewards_[i*batch:i*batch+batch]
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
                
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
                print(old_states.grad)
                self.optimizer.step()
                
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
        
        
       



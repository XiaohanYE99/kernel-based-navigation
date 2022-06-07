import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
from robot_envs.sparse_solver import *#SparseSolve
import time
plt.ion()

import time
import torch
import taichi as ti
import math
import random
from numba import njit, prange
import copy
import heapq
from torch.autograd import Variable
import torch.nn.functional as F
# extract some functions for easy calling

pi=np.pi
def find_grid_index(pos,dx):
    return int(pos[1]/dx)*100+int(pos[0]/dx)
#@njit(parallel=True)  
def P2G(state,state_p,n_robots,dx):
    for i in prange(n_robots):
        x=int(state[i*2]//dx)
        y=int(state[i*2+1]//dx)
        state_p[0][x][y]=1.0
    return state_p

def get_angle(v1, v2):
  dx1 = v1[0]
  dy1 = v1[1]
  dx2 = v2[0]
  dy2 = v2[1]
  angle1 = math.atan2(dy1, dx1)
  angle1 = int(angle1 * 180/math.pi)
  # print(angle1)
  angle2 = math.atan2(dy2, dx2)
  angle2 = int(angle2 * 180/math.pi)
  # print(angle2)
  if angle1*angle2 >= 0:
    included_angle = abs(angle1-angle2)
  else:
    included_angle = abs(angle1) + abs(angle2)
    if included_angle > 180:
      included_angle = 360 - included_angle
  return included_angle
def dist(pos,aim):
        dis=pos-aim
        return np.sqrt(dis[0]*dis[0]+dis[1]*dis[1])
def getdis(p1,p2,p):
    a=p2[1]-p1[1]
    b=p1[0]-p2[0]
    c=p2[0]*p1[1]-p1[0]*p2[1]
    return math.fabs(a*p[0]+b*p[1]+c)/pow(a*a+b*b,0.5)
def isIn(p1,p2,p,radius):
    if dist(p1,p)<radius:
        return 1
    if dist(p2,p)<radius:
        return 1
    if getdis(p1,p2,p)<radius:
        return 1
    return 0      
def get_reward(n_robots,aim,state,oldstate,angle,dis,dx):
        #aim reward
        reward1=0
        reward2=0
        dis1=0.0
        dis2=0.0
        num=0.0
        for i in prange(n_robots):
            d=state[i*2:i*2+2]-np.array(aim)
            if math.sqrt(d[0]*d[0]+d[1]*d[1])<0.19:
                num+=0.01
            else:
                idx_old=int(oldstate[i*2+1]/dx)*100+int(oldstate[i*2]/dx)
                idx_now=int(state[i*2+1]/dx)*100+int(state[i*2]/dx)
                
                dis1=dis[idx_old]
                dis2=dis[idx_now]
                
                if dis1>dis2:
                    reward1+=4.0*(dis1-dis2)
                else:
                    reward1+=6.0*(dis1-dis2)
                    
            if angle[i]>45:
                reward2-=(1.5e-4)*angle[i]
        #collision reward
        reward1+=pow(num,1)
        #reward1+=400.0*(dis1-dis2)
        '''
        for i in prange(n_robots):
            for j in prange(i+1,n_robots):
                dis=state[i*2:i*2+2]-state[j*2:j*2+2]
                if np.sqrt(dis[0]*dis[0]+dis[1]*dis[1])<2*radius:
                    reward-=0.3
            
            
            for j in prange(bdset.shape[0]):
                p1=np.array([bdset[j][0],bdset[j][1]])
                p2=np.array([bdset[j][2],bdset[j][3]])
                p=state[i*2:i*2+2]
                dis1=p1-p
                dis2=p2-p
                a=p2[1]-p1[1]
                b=p1[0]-p2[0]
                c=p2[0]*p1[1]-p1[0]*p2[1]
                d=math.fabs(a*p[0]+b*p[1]+c)/pow(a*a+b*b,0.5)
                if np.sqrt(dis1[0]*dis1[0]+dis1[1]*dis1[1])<radius or np.sqrt(dis2[0]*dis2[0]+dis2[1]*dis2[1])<radius or d<radius:
                    reward-=0.3
               '''
        if abs(reward1)>10:
            reward1=0.0
        return reward1,reward2
def dijkstra(dx,start,bdset):
    x=1.5*dx
    y=1.5*dx
    G={}
    L=[]
    L.append([0,dx,0.1])
    L.append([0,-dx,0.1])
    L.append([dx,0,0.1])
    L.append([-dx,0,0.1])
    L.append([dx,dx,0.1414])
    L.append([dx,-dx,0.1414])
    L.append([-dx,dx,0.1414])
    L.append([-dx,-dx,0.1414])
    for i in range(40000):
        G[i]={}
    while x<0.99:
        y=1.5*dx
        while y<0.99:
            idx=find_grid_index([x,y],dx)
            for i in range(8):
                flag=1
                tx=x+L[i][0]
                ty=y+L[i][1]
                for j in range(len(bdset)):
                    if tx>=bdset[j][0] and tx<=bdset[j][2] and ty>=bdset[j][1] and ty<=bdset[j][3]:
                        #print(tx,ty)
                        flag=0
                if flag==1:
                    G[idx][find_grid_index([tx,ty],dx)]=L[i][2]
            y+=dx
        x+=dx
    INF = 999999999
 
    dis = dict((key,INF) for key in G) 
    dis[start] = 0
    vis = dict((key,False) for key in G) 


    pq = []  
    heapq.heappush(pq,[dis[start],start])
 
    t3 = time.time()
    path = dict((key,[start]) for key in G) 
    while len(pq)>0:
        v_dis,v = heapq.heappop(pq)  
        if vis[v] == True:
            continue
        vis[v] = True
        p = path[v].copy()   
        for node in G[v]:   

            new_dis = dis[v] + float(G[v][node])
            if new_dis < dis[node] and (not vis[node]): 
                dis[node] = new_dis 

                heapq.heappush(pq,[dis[node],node])
                temp = p.copy()
                temp.append(node)
                path[node] = temp   
 
    t4 = time.time()

    return dis

class NavigationEnvs():
    def __init__(self,gui,sim,use_kernel_loop,use_sparse_FEM):
        self.sim=sim
        self.gui=gui
        self.use_kernel_loop=use_kernel_loop
        self.use_sparse_FEM=use_sparse_FEM
        
        self.sim.setNewtonParameters(100,1e-0,5e-4,5e4,1e-6)

        self.N=20           #kernel number
        self.radius=0.01    #robot radius
        self.n_robots=100   #robot number
        self.eps=0.02
        self.agent=[]
        self.suc=0
        self.arrive=0
        self.over1=0
        self.device = torch.device("cuda:0")

        
        self.state=np.zeros([self.n_robots*2])
        self.oldstate=np.zeros([self.n_robots*2])
        self.vel=np.zeros([self.n_robots*2])
        self.oldvel=np.zeros([self.n_robots*2])
        self.angle=np.zeros([self.n_robots])
        self.size=100
        self.size_x=100
        self.size_y=100
        self.dx=1.0/self.size_x
        self.aim=[0.7,0.5]
        self.goal=np.zeros([self.n_robots*2])
        self.begin=np.zeros([self.n_robots*2])
        self.t=np.zeros([self.n_robots])

        self.tot_num=0
        self.cnt=0
        #kernel parameters
        self.alpha=np.zeros(self.N)
        self.omega=np.zeros(self.N)
        self.x0=np.zeros(self.N)
        self.y0=np.zeros(self.N)
        self.noactive=np.zeros(self.n_robots)
        self.vis=np.zeros(self.n_robots)
        self.deltap=np.zeros([self.n_robots,2])
        
        #FEM
        self.grid_ux=(torch.arange(0.0,self.size_x+1)*self.dx).unsqueeze(1).expand(self.size_x+1,self.size_y).to(self.device)
        self.grid_uy=(torch.arange(0.5,self.size_y)*self.dx).unsqueeze(0).expand(self.size_x+1,self.size_y).to(self.device)
        self.grid_vx=(torch.arange(0.5,self.size_x)*self.dx).unsqueeze(1).expand(self.size_x,self.size_y+1).to(self.device)
        self.grid_vy=(torch.arange(0.0,self.size_y+1)*self.dx).unsqueeze(0).expand(self.size_x,self.size_y+1).to(self.device)
        #self.GT=torch.zeros([self.size_x*self.size_y,self.size_x*(self.size_y+1)+self.size_y*(self.size_x+1)],dtype=torch.float32)#.to(self.device)
        #self.G=torch.zeros([self.size_x*(self.size_y+1)+self.size_y*(self.size_x+1),self.size_x*self.size_y],dtype=torch.float32).to(self.device)
        #self.L=torch.zeros([self.size_x*self.size_y,self.size_x*self.size_y],dtype=torch.float32).to(self.device)
        #self.V=torch.zeros([self.size_x*(self.size_y+1)+self.size_y*(self.size_x+1),1],dtype=torch.float32).to(self.device)
        
        self.mask_x=torch.ones([self.size_x+1,self.size_y],dtype=torch.float32).to(self.device)
        self.mask_y=torch.ones([self.size_x,self.size_y+1],dtype=torch.float32).to(self.device)
        
        self.pos_reward=0
        self.vel_reward=0
        
        self.bdset=[[0.5,0.2,0.54,0.8],[0.2,0.2,0.54,0.24],[0.2,0.76,0.54,0.8],
                    [0.0,0.0,1.0,0.04],[0.0,0.96,1.0,1.0],[0.0,0.0,0.04,1.0],[0.96,0.0,1.0,1.0]]#,[0.95,0.52,1.0,1.0]]
        
        self.observation_space=np.zeros(2*self.n_robots)
        self.action_space=np.zeros(2*self.n_robots)#(4*self.N)
        #print(self.boundary.N)
        
       
        self.init_state=[]
        
        for i in range(4):
            self.init_state.append([0.06+0.1*i,0.07])
            self.init_state.append([0.06+0.1*i,0.82])
            
        for i in range(2):
            for j in range(4):
                self.init_state.append([0.2+0.1*i,0.3+0.1*j])
        
        for i in range(5):
            for j in range(20):
                idx=i*20+j

                self.state[idx*2]=i*0.02+0.35
                self.state[idx*2+1]=j*0.02+0.3
                self.oldstate[idx*2]=i*0.02+0.35
                self.oldstate[idx*2+1]=j*0.02+0.3
                self.agent.append(self.sim.addAgent((self.state[idx*2], self.state[idx*2+1]), 0.04, 100, 0.04, 0.04, 0.009, 1, (0, 0)))
                
        self.sim.addObstacle([(0.2, 0.2), (0.54, 0.2), (0.54, 0.8),(0.2,0.8),(0.2,0.76),(0.5,0.76),(0.5,0.24),(0.2,0.24)])
        self.sim.addObstacle([(0.04, 0.04),(0.04,0.96),(0.96,0.96),(0.96,0.04) ])
        self.sim.processObstacles()
        self.dis=dijkstra(self.dx,find_grid_index([0.7,0.5],self.dx),self.bdset)
        self.FEM_init()
        self.sparsesolve = SparseSolve.apply
        torch.cuda.empty_cache()

    def reset(self):
        self.suc=0
        self.cnt=0
        self.div=np.ones(self.n_robots)
        self.noactive=np.zeros(self.n_robots)
        self.vis=np.zeros(self.n_robots)
        o=random.sample(range(0, 15), 4)
        o.sort()
        
        idx=0
        for k in o:
            for i in range(5):
                for j in range(5):
                    self.t[idx]=40
                    self.state[idx*2:idx*2+2]=[0.02*i+self.init_state[k][0],0.02*j+self.init_state[k][1]]
                    self.oldstate[idx*2:idx*2+2]=[0.02*i+self.init_state[k][0],0.02*j+self.init_state[k][1]]
                    #self.agent.append(self.sim.addAgent((self.state[idx*2], self.state[idx*2+1]), 0.04, 5, 0.04, 0.04, 0.01, 2, (0, 0)))
                    self.sim.setAgentPosition(self.agent[i],(self.state[idx*2], self.state[idx*2+1]))
                    idx+=1
        '''
        for i in range(50):
            for j in range(50):
                idx=int(50*i+j)
                self.state[idx*2:idx*2+2]=np.array([0.02*i,0.02*j+0.01])
        for i in range(50):
            for j in range(50):
                idx=int(50*i+j+2500)
                self.state[idx*2:idx*2+2]=np.array([0.02*i+0.01,0.02*j])
            '''
        
        state_p=np.zeros([1,self.size_x,self.size_y])
        state_p=P2G(self.state,state_p,self.n_robots,self.dx)
        return self.state#np.append(self.state,np.zeros(2*self.n_robots))
    
    def find_grid_index(self):
        pos=self.state.reshape([self.n_robots,2])
        self.in_grid=(pos[:,1]/self.dx).astype(np.int32)*self.size+(pos[:,0]/self.dx).astype(np.int32)
    def render(self):
        pos=self.state.reshape([-1,2])
        #bd=self.aim.reshape([-1,2])
        q=np.zeros([self.N,2])
        for i in range(self.N):
            q[i]=[self.x0[i],self.y0[i]]
        #self.gui.circles(q, radius=5, color=0x068587)
        self.gui.circles(pos, radius=4.5, color=0xF7EED6)
        #self.gui.circles(bd, radius=2, color=0xFF0000)
        self.gui.rect([0.5, 0.2], [0.54, 0.8], radius=1, color=0xF4A460)
        self.gui.rect([0.2, 0.2], [0.5, 0.24], radius=1, color=0xF4A460)
        self.gui.rect([0.2, 0.76], [0.5, 0.8], radius=1, color=0xF4A460)
        self.gui.lines(begin=pos, end=pos+0.04*self.deltap, radius=1, color=0x068587)
        #self.gui.show('img/MAC.png')
        self.gui.show()
    def FEM_init(self):
        mask=torch.zeros([self.size_x,self.size_y])
        GT=torch.zeros([self.size_x*self.size_y,self.size_x*(self.size_y+1)+self.size_y*(self.size_x+1)],dtype=torch.float32)#.to(self.device)
        for i in range(self.size_x):
            for j in range(self.size_y):
                for k in range(len(self.bdset)):
                    if (i+0.5)*self.dx>=self.bdset[k][0] and (i+0.5)*self.dx<=self.bdset[k][2] and (j+0.5)*self.dx>=self.bdset[k][1] and (j+0.5)*self.dx<=self.bdset[k][3]:
                        mask[i,j]=1
                    '''
                    if (i)*self.dx>=self.bdset[k][0] and (i)*self.dx<=self.bdset[k][2] and (j+0.5)*self.dx>=self.bdset[k][1] and (j+0.5)*self.dx<=self.bdset[k][3]:
                        self.mask_x[i,j]=0
                    if (i+0.5)*self.dx>=self.bdset[k][0] and (i+0.5)*self.dx<=self.bdset[k][2] and (j)*self.dx>=self.bdset[k][1] and (j)*self.dx<=self.bdset[k][3]:
                        self.mask_y[i,j]=0  
                        '''
        for i in range(self.size_x):
            for j in range(self.size_y):
                now=j*self.size_x+i
                left=j*(self.size_x+1)+i
                right=j*(self.size_x+1)+i+1
                up=(j+1)*self.size_x+i+(self.size_x+1)*self.size_y
                down=j*self.size_x+i+(self.size_x+1)*self.size_y
                GT[now][left]=1
                GT[now][right]=-1
                GT[now][up]=-1
                GT[now][down]=1
                
                if i>0 and j>0 and i<self.size_x-1 and j<self.size_y-1:
                    if mask[i-1,j]!=mask[i,j]: GT[now][left]=0
                    if mask[i+1,j]!=mask[i,j]: GT[now][right]=0
                    if mask[i,j-1]!=mask[i,j]: GT[now][down]=0
                    if mask[i,j+1]!=mask[i,j]: GT[now][up]=0
                
                
                if mask[i,j]==1:
                    self.mask_x[i,j]=0
                    self.mask_x[i+1,j]=0
                    self.mask_y[i,j]=0
                    self.mask_y[i,j+1]=0
        
        self.GT=GT.to_sparse().to(self.device)
        
        self.G=GT.T.to(self.device)
        

        del(GT)
        
        alpha=2.5e0*(self.dx*self.dx)
        if self.use_sparse_FEM:
        
            I=torch.eye(self.size_x*self.size_y,self.size_x*self.size_y,dtype=torch.float32).to_sparse().to(self.device)
            self.L=(torch.matmul(self.GT,self.G)+alpha*I).to_sparse()
        
        else:
            I=torch.eye(self.size_x*self.size_y,self.size_x*self.size_y,dtype=torch.float32).to(self.device)
            self.L=(torch.matmul(self.GT,self.G)+alpha*I).inverse()
            
        self.G=self.G.to_sparse()
        
        del(mask)
        '''
        self.L=(torch.matmul(self.GT,self.G)+alpha*I).to_sparse()

        self.G=self.G.to_sparse()
        self.GT=self.GT.to_sparse()
        '''
        
    def P2G(self,pos):
        k=pos.size(0)
        px=pos[:,::2]
        py=pos[:,1::2]
        idx_x=px//self.dx
        idx_y=py//self.dx
        i = torch.empty([3,self.n_robots]).long().to(self.device)

        i[0]=torch.arange(k).repeat_interleave(self.n_robots)
        i[1]=torch.flatten(idx_x)
        i[2]=torch.flatten(idx_x)
        v=torch.ones([k,self.n_robots])
        state=torch.sparse.FloatTensor(i, v, torch.Size([k,self.size_x,self.size_y]))#.to(self.device)
        return state.to_dense()
    def projection(self,action):

        #action=torch.rand(action.size()).to(self.device)
        '''
        for i in range(10):
            action[:,i*4]=0.1+0.02*i
            action[:,i*4+1]=0.2
            action[:,i*4+2]=0.2
            action[:,i*4+3]=1
        for i in range(10,20):
            action[:,i*4]=-0.1+0.02*i
            action[:,i*4+1]=0.8
            action[:,i*4+2]=0.2
            action[:,i*4+3]=0
            '''
        
        #print(action.requires_grad)
        t0=time.time()
        k=action.size(0)
        
        x0=action[:,::4].unsqueeze(2).unsqueeze(3)
        y0=action[:,1::4].unsqueeze(2).unsqueeze(3)
        alpha=(action[:,2::4].unsqueeze(2).unsqueeze(3))*19.0+1.0
        omega=action[:,3::4].unsqueeze(2).unsqueeze(3)
        #alpha[alpha<=1]=1
        if self.use_kernel_loop==False:
            ux=self.grid_ux.unsqueeze(0).unsqueeze(0)
            uy=self.grid_uy.unsqueeze(0).unsqueeze(0)

            r=torch.sqrt(torch.pow(x0-ux,2)+torch.pow(y0-uy,2))
            
            velocity_x=(torch.sum(-4.0*(2*omega-1)*torch.exp(-alpha*r)*(uy-y0)/r,dim=1))*self.mask_x
            
            vx=self.grid_vx.unsqueeze(0).unsqueeze(0)
            vy=self.grid_vy.unsqueeze(0).unsqueeze(0)
            
            r=torch.sqrt(torch.pow(vx-x0,2)+torch.pow(vy-y0,2))
            velocity_y=(torch.sum(4.0*(2*omega-1)*torch.exp(-alpha*r)*(vx-x0)/r,dim=1))*self.mask_y
        else:
            ux=self.grid_ux.unsqueeze(0)
            uy=self.grid_uy.unsqueeze(0)
            vx=self.grid_vx.unsqueeze(0)
            vy=self.grid_vy.unsqueeze(0)
            velocity_x=torch.zeros(k,self.N,self.size_x,self.size_y)
            velocity_y=torch.zeros(k,self.N,self.size_x,self.size_y)
            for i in range(self.N):
                r=torch.sqrt(torch.pow(ux-x0[:,i,:,:],2)+torch.pow(uy-y0[:,i,:,:],2))
                velocity_x+=(torch.sum(-4.0*(2*omega[:,i,:,:]-1)*torch.exp(-alpha[:,i,:,:]*r)*(uy-y0[:,i,:,:])/r,dim=1))*self.mask_x
            for i in range(self.N):
                r=torch.sqrt(torch.pow(vx-x0[:,i,:,:],2)+torch.pow(vy-y0[:,i,:,:],2))
                velocity_y+=(torch.sum(4.0*(2*omega[:,i,:,:]-1)*torch.exp(-alpha[:,i,:,:]*r)*(vx-x0[:,i,:,:])/r,dim=1))*self.mask_y
        
        

        velocity=torch.cat((torch.flatten(velocity_x.transpose(1,2),1),torch.flatten(velocity_y.transpose(1,2),1)),1)#.unsqueeze(2)
        
        
        if self.use_sparse_FEM==False:
            velocity=(velocity).T
            v1=torch.matmul(self.GT,velocity)
            v2=torch.matmul(self.L,v1)
            v3=torch.matmul(self.G,v2)
            velocity=velocity-v3
            velocity=(velocity.T).contiguous().view(k,-1)
        
        
        else:
            velocity=(velocity).T
            v1=torch.matmul(self.GT,velocity)
            
            v2=self.sparsesolve(self.L,v1)

            v3=torch.matmul(self.G,v2)
            velocity=velocity-v3
            velocity=(velocity.T).contiguous().view(k,-1)
        
        
        #print((torch.sum(torch.abs(torch.matmul(self.GT,velocity)))))
        #velocity=torch.matmul(self.P,velocity)
        #print(time.time()-t0)
        return velocity#.squeeze(2)
    
    def get_velocity(self,pos,velocity):
        #print(velocity.size())
        t0=time.time()
        k=pos.size(0)
        
        #print((torch.sum(torch.abs(torch.matmul(self.GT,velocity)))))
        px=pos[:,::2]
        py=pos[:,1::2]
        
        idx_x=torch.floor(px/self.dx)
        idx_y=torch.floor(py/self.dx)
        alpha_x=px/self.dx-idx_x
        alpha_y=py/self.dx-idx_y
        
        velocity_x=velocity[:,:(self.size_x+1)*self.size_y]
        velocity_y=velocity[:,(self.size_x+1)*self.size_y:]
        idx1=(idx_y*(self.size_x+1)+idx_x).long()
        vel_x=torch.gather(velocity_x,1,idx1)*(1.0-alpha_x)+torch.gather(velocity_x,1,idx1+1)*alpha_x
        idx2=(idx_y*self.size_x+idx_x).long()
        vel_y=torch.gather(velocity_y,1,idx2)*(1.0-alpha_y)+torch.gather(velocity_y,1,idx2+self.size_x)*alpha_y

        
        
        #velocity boundary
        rr=torch.sqrt(vel_x*vel_x+vel_y*vel_y)
        rr=F.relu(rr/2.0-1.0)+1.0
        
        #print(time.time()-t0)
        #print(vel_y.size())
        return torch.cat((vel_x/rr,vel_y/rr),1)#.squeeze(2)
    def step(self,velocity):
        #action=action*0.5+0.5
        velocity=velocity.squeeze(0)

        '''
        action=action.squeeze(0)
        t0=time.time()
        velocity=self.projection(action)
        
        for i in range(self.N):
            self.x0[i]=action[4*i]#0~1
            self.y0[i]=action[4*i+1]#0~1
        #print(action)
        '''
        
        for i in range(self.n_robots):
            self.oldstate[i*2:i*2+2]=self.state[i*2:i*2+2]
        state_p=np.zeros([1,self.size_x,self.size_y])
        #self.BEM_solver()
        vx=velocity[:self.n_robots]
        vy=velocity[self.n_robots:]
        vx=vx.cpu().detach().numpy()
        vy=vy.cpu().detach().numpy()
        for it in range(2):
            pos=self.state.reshape([-1,2])

            for i in range(self.n_robots):
                if i<self.n_robots/2:
                    self.deltap[i]=[vx[i]*0.1,vy[i]*0.0]
                else:
                    self.deltap[i]=[vx[i]*0.0,vy[i]*0.1]
                
            #print(time.time()-t0)
            
            for i in range(self.n_robots):
                dx=vx[i]
                dy=vy[i]
                #print(dx,dy)
                lenn=math.sqrt(dx*dx+dy*dy)
                if lenn>2.0:
                    dx*=2.0/lenn
                    dy*=2.0/lenn
                
                if self.state[i*2]>0.51 and self.state[i*2]<0.89 and self.state[i*2+1]<0.69 and self.state[i*2+1]>0.31:
                    self.t[i]-=1
                    r=np.sqrt((pow(pos[i][0]-0.7,2)+pow(pos[i][1]-0.5,2)))
                    dx=1.0*(0.7-pos[i][0])/r
                    dy=1.0*(0.5-pos[i][1])/r
                    if r<0.01 or self.t[i]<=0:
                        dx=0#*=r/0.01
                        dy=0#*=r/0.01
                self.sim.setAgentPrefVelocity(self.agent[i], (dx, dy))
                self.sim.setAgentPosition(self.agent[i], (pos[i][0], pos[i][1]))
                self.deltap[i]=[dx,dy]

            self.sim.doNewtonStep(False)

            for i in range(self.n_robots):
                self.state[i*2:i*2+2]=self.sim.getAgentPosition(self.agent[i])
        for i in range(self.n_robots):
            self.oldvel[i*2:i*2+2]=self.vel[i*2:i*2+2]
            self.vel[i*2:i*2+2]=self.state[i*2:i*2+2]-self.oldstate[i*2:i*2+2]
            self.angle[i]=get_angle(self.vel[i*2:i*2+2],self.oldvel[i*2:i*2+2])
            
        reward1,reward2=get_reward(self.n_robots,self.aim,self.state,self.oldstate,self.angle,self.dis,self.dx)
        reward=reward1#+reward2
        self.pos_reward+=reward1
        self.vel_reward+=reward2
        done=0
        #state_p=P2G(self.state,state_p,self.n_robots,self.dx)
        #self.render(0)
        #print(time.time()-t0)
        '''
        a=np.append(self.x0,self.y0)
        b=np.append(self.omega,self.alpha)
        c=np.append(a,b)
        '''
        if self.cnt%1==0:
            self.render()
        self.cnt+=1
        return self.state,reward,done,dict(reward=reward)

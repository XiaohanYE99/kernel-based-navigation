import numpy as np

import scipy as sp
import scipy.sparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
import time
from torch.autograd import Function

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

from robot_envs.sparse_solver import SparseSolve
from robot_envs.viewer import Viewer

# extract some functions for easy calling

pi = np.pi


def find_grid_index(pos, dx):
    return int((pos[1]+0.1*dx) / dx) * 100 + int((pos[0]+0.1*dx) / dx)


def make_ccw(pts):
    is_ccw = False
    for i in range(len(pts)):
        iLast = (i + len(pts) - 1) % len(pts)
        dirLast = (pts[i][0] - pts[iLast][0], pts[i][1] - pts[iLast][1])

        iNext = (i + 1) % len(pts)
        dirNext = (pts[iNext][0] - pts[i][0], pts[iNext][1] - pts[i][1])

        nLast = (-dirLast[1], dirLast[0])
        dotLastNext = nLast[0] * dirNext[0] + nLast[1] * dirNext[1]
        if dotLastNext > 0:
            is_ccw = True
        elif dotLastNext < 0:
            is_ccw = False
            break

    if not is_ccw:
        return [pts[len(pts) - 1 - i] for i in range(len(pts))]
    else:
        return pts




def get_reward(n_robots, aim, state, oldstate, angle, dis, dx):
    # aim reward
    reward1 = 0
    reward2 = 0
    dis1 = 0.0
    dis2 = 0.0
    num = 0.0
    for i in prange(n_robots):
        d = state[i * 2:i * 2 + 2] - np.array(aim)
        if math.sqrt(d[0] * d[0] + d[1] * d[1]) < 0.19:
            num += 0.01
        else:
            idx_old = int(oldstate[i * 2 + 1] / dx) * 100 + int(oldstate[i * 2] / dx)
            idx_now = int(state[i * 2 + 1] / dx) * 100 + int(state[i * 2] / dx)

            dis1 = dis[idx_old]
            dis2 = dis[idx_now]

            if dis1 > dis2:
                reward1 += 4.0 * (dis1 - dis2)
            else:
                reward1 += 6.0 * (dis1 - dis2)
    # collision reward
    reward1 += pow(num, 1)
    # reward1+=400.0*(dis1-dis2)
    if abs(reward1) > 10:
        reward1 = 0.0
    return reward1, reward2
def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in

#@njit(parallel=True)
def obstacleMap(grid_size,dx,obstacles,wind_size):
    obs_map=torch.zeros(grid_size)
    for i in prange(1,grid_size[0]):
        for j in prange(1,grid_size[1]):
            pos0=[(i)*dx,(j)*dx]
            for obs in obstacles:
                if is_in_poly(pos0,obs/wind_size):
                    obs_map[i,j]+=0.25
                    obs_map[i-1,j]+=0.25
                    obs_map[i,j-1]+=0.25
                    obs_map[i-1,j-1]+=0.25
                    break

    return obs_map

def dijkstra(dx, start, obs_map):
    x = dx
    y = dx
    G = {}
    L = []
    L.append([0, dx, 0.1])
    L.append([0, -dx, 0.1])
    L.append([dx, 0, 0.1])
    L.append([-dx, 0, 0.1])
    L.append([dx, dx, 0.1414])
    L.append([dx, -dx, 0.1414])
    L.append([-dx, dx, 0.1414])
    L.append([-dx, -dx, 0.1414])
    for i in range(12000):
        G[i] = {}
    while x < 1.0-dx:
        y = dx
        while y < 1.0-dx:
            idx = find_grid_index([x, y], dx)

            for i in range(8):
                flag = 1
                tx = x + L[i][0]
                ty = y + L[i][1]
                ix=int(tx/dx)
                iy=int(ty/dx)
                if obs_map[ix,iy]<1 or ix<=0.02/dx or ix>=0.98/dx or iy<=0.02/dx or iy>=0.98/dx:
                    G[idx][find_grid_index([tx, ty], dx)] = L[i][2]
            y += dx
        x += dx
    INF = 999999999

    dis = dict((key, INF) for key in G)
    dis[start] = 0
    vis = dict((key, False) for key in G)

    pq = []
    heapq.heappush(pq, [dis[start], start])

    t3 = time.time()
    path = dict((key, [start]) for key in G)
    while len(pq) > 0:
        v_dis, v = heapq.heappop(pq)
        if vis[v] == True:
            continue
        vis[v] = True
        p = path[v].copy()
        for node in G[v]:

            new_dis = dis[v] + float(G[v][node])
            if new_dis < dis[node] and (not vis[node]):
                dis[node] = new_dis

                heapq.heappush(pq, [dis[node], node])
                temp = p.copy()
                temp.append(node)
                path[node] = temp

    diss = list(dis.values())
    return torch.tensor(diss, dtype=torch.float32)


class NavigationEnvs():
    def __init__(self, batch_size,gui, sim, multisim,use_kernel_loop, use_sparse_FEM):

        self.batch_size=batch_size
        self.sim = sim
        self.multisim=multisim
        self.gui = gui
        self.use_kernel_loop = use_kernel_loop
        self.use_sparse_FEM = use_sparse_FEM
        self.current_obs=[]
        self.viewer=None

        self.N = 15  # kernel number
        self.radius = 0.008  # robot radius
        self.n_robots = 50  # robot number
        self.scale=250

        self.agent = []
        self.suc = 0
        self.arrive = 0
        self.over1 = 0
        self.device = torch.device("cuda:0")
        #self.device = torch.device("cpu")

        self.state = np.zeros([self.n_robots * 2])
        self.oldstate = np.zeros([self.n_robots * 2])
        self.vel = np.zeros([self.n_robots * 2])
        self.oldvel = np.zeros([self.n_robots * 2])
        self.angle = np.zeros([self.n_robots])
        self.size = 100
        self.size_x = 100
        self.size_y = 100
        self.dx = 1.0 / self.size_x
        self.aim = [0.9, 0.5]
        self.goal = np.zeros([self.n_robots * 2])
        self.begin = np.zeros([self.n_robots * 2])
        self.bound=0.1
        self.obs_map=torch.zeros([self.size_x,self.size_y])
        self.target_map=torch.zeros([self.size_x,self.size_y])

        self.tot_num = 0
        self.cnt = 0
        self.task_id=0
        # kernel parameters
        self.alpha = np.zeros(self.N)
        self.omega = np.zeros(self.N)
        self.x0 = np.zeros(self.N)
        self.y0 = np.zeros(self.N)
        self.noactive = np.zeros(self.n_robots)
        self.vis = np.zeros(self.n_robots)
        self.deltap = np.zeros([self.n_robots, 2])
        self.eps = 1e-4
        self.target = 0
        self.init_state = []

        # FEM
        self.grid_ux = (torch.arange(0.0, self.size_x + 1) * self.dx).unsqueeze(1).expand(self.size_x + 1,
                                                                                          self.size_y).to(self.device)
        self.grid_uy = (torch.arange(0.5, self.size_y) * self.dx).unsqueeze(0).expand(self.size_x + 1, self.size_y).to(
            self.device)
        self.grid_vx = (torch.arange(0.5, self.size_x) * self.dx).unsqueeze(1).expand(self.size_x, self.size_y + 1).to(
            self.device)
        self.grid_vy = (torch.arange(0.0, self.size_y + 1) * self.dx).unsqueeze(0).expand(self.size_x,
                                                                                          self.size_y + 1).to(
            self.device)

        self.mask_x = torch.ones([self.size_x + 1, self.size_y], dtype=torch.float32).to(self.device)
        self.mask_y = torch.ones([self.size_x, self.size_y + 1], dtype=torch.float32).to(self.device)

        self.observation_space = np.zeros(2 * self.n_robots)
        self.action_space = np.zeros(5 * self.N)  # (4*self.N)

        for i in range(self.n_robots):
            self.agent.append(
                self.sim.addAgent(np.array([0.,0.],dtype=float),np.array([0.,0.])))

        for j in range(self.n_robots):
            pos, vel = [], []
            for i in range(self.batch_size):
                pos.append(np.array([random.randrange(-self.n_robots, self.n_robots), random.randrange(-self.n_robots, self.n_robots)], dtype=float))
                vel.append(np.array([0., 0.]))
            self.multisim.addAgent(pos,vel)

        self.sparsesolve = SparseSolve.apply
        self.a=0


    def load_roadmap(self,fn):
        import pickle
        current_obs, wind_size, _, _ = pickle.load(open(fn, 'rb'), encoding='iso-8859-1')

        current_obs.append(Viewer.get_box_ll(x=675, y=12, lowerleft=(0, 0)))
        current_obs.append(Viewer.get_box_ll(x=12, y=675, lowerleft=(663, 0)))
        current_obs.append(Viewer.get_box_ll(x=675, y=12, lowerleft=(0, 663)))
        current_obs.append(Viewer.get_box_ll(x=12, y=675, lowerleft=(0, 0)))
        self.wind_size = wind_size
        if current_obs is not None:
            self.sim.clearObstacle()
            self.multisim.clearObstacle()
            self.current_obs = []
            for obs in current_obs:
                self.current_obs.append(obs)
                obb = []
                for p in obs:
                    obb.append(np.array(p * np.array([self.scale, self.scale]) / np.array(wind_size),dtype=float))

                self.sim.addObstacle(obb)
                self.multisim.addObstacle(obb)

        self.obs_map = obstacleMap([self.size_x, self.size_y], self.dx, self.current_obs, np.array(self.wind_size))
        self.reset()
        self.FEM_init()
        torch.cuda.empty_cache()

    def reset_viewer(self):
        for i in range(self.n_robots):
            self.viewer.add_agent((self.state[i]*self.wind_size[0]/self.scale, self.state[i+self.n_robots]*self.wind_size[1]/self.scale), self.radius*self.wind_size[0])
        for i in range(self.N):
            self.viewer.add_waypoint((self.x0[i]*self.wind_size[0],self.y0[i]*self.wind_size[0]),self.radius*self.wind_size[0])
        #for p in self.goal_positions:
        self.viewer.add_goal((self.aim[0]*self.wind_size[0], self.aim[1]*self.wind_size[0]), 0.016*self.wind_size[0])
        for obs in self.current_obs:
            obs_toadd = []
            for points in obs:
                obs_toadd += [points[0], points[1]]
            self.viewer.add_obs(obs_toadd)
        if hasattr(self, 'sensor'):
            self.viewer.sensor = self.sensor

    def render(self):
        '''
        if self.cnt%10==0:
            path='./robot_envs/mazes_g100w700h700/maze'+str(self.task_id)+'.dat'
            self.load_roadmap(path)
            self.task_id+=1
        '''

        if self.viewer is None:
            self.viewer = Viewer(wind_size=(675,675))
            self.viewer.env = self
            self.reset_viewer()

        for i in range(self.n_robots):
            self.viewer.agent_pos_array[i] = (self.state[i]*self.wind_size[0]/self.scale, self.state[i+self.n_robots]*self.wind_size[1]/self.scale)
        for i in range(self.N):
            self.viewer.waypoint_pos_array[i]=(self.x0[i]*self.wind_size[0],self.y0[i]*self.wind_size[0])
        self.viewer.goal_pos_array[0] = (self.aim[0]*self.wind_size[0], self.aim[1]*self.wind_size[0])
        self.viewer.render()
    def reset_init_agent(self):
        unit=1.0
        grid=7.0
        self.init_state=[]
        x=0.5*unit/grid
        end=6.5*unit/grid
        while x<end:
            y = 0.5 * unit / grid
            while y<end:
                idx,idy=int(x/self.dx),int(y/self.dx)
                if self.obs_map[idx,idy]==0 and self.obs_map[idx,idy+1]==0 and self.obs_map[idx+1,idy+1]==0 and self.obs_map[idx+1,idy]==0 and self.obs_map[idx+1,idy-1]==0\
                        and self.obs_map[idx,idy-1]==0 and self.obs_map[idx-1,idy-1]==0 and self.obs_map[idx-1,idy]==0 and self.obs_map[idx-1,idy+1]==0:
                    self.init_state.append([x,y])
                y+=2.5*self.radius
            x+=2.5*self.radius
        #self.reset_agent()
        self.reset_aim()
    def reset_agent(self):
        agent_no=random.sample(self.init_state, self.n_robots)
        for i in range(self.n_robots):
            self.state[i] = agent_no[i][0]
            self.state[i + self.n_robots] = agent_no[i][1]
        return self.state*self.scale
    def reset_aim(self):
        p=np.random.rand()*0.8+0.1
        self.a=(self.a+1)%4
        a=self.a
        if a==0:
            self.aim=[p,0.05]
        elif a==1:
            self.aim=[1.0-p,0.95]
        elif a==2:
            self.aim=[0.05,1.0-p]
        else:
            self.aim=[0.95,p]
        #self.aim=[0.05,0.05]
    def reset(self):
        self.reset_init_agent()
        self.dis = dijkstra(self.dx, find_grid_index(self.aim, self.dx), self.obs_map).to(self.device)
        self.dis[self.dis>100]=20

        self.get_target_map()

        if self.viewer is not None:
            self.viewer.reset_array()
            self.reset_viewer()
        #return self.state  # np.append(self.state,np.zeros(2*self.n_robots))
    '''
    def find_grid_index(self):
        pos = self.state.reshape([self.n_robots, 2])
        self.in_grid = (pos[:, 1] / self.dx).astype(np.int32) * self.size + (pos[:, 0] / self.dx).astype(np.int32)
    '''
    def get_target_map(self):
        for i in range(self.size_x):
            for j in range(self.size_y):
                self.target_map[i,j]=self.dis[int(j*self.size_y+i)]
    def FEM_init(self):

        mask = torch.zeros([self.size_x, self.size_y])
        GT = torch.zeros([self.size_x * self.size_y, self.size_x * (self.size_y + 1) + self.size_y * (self.size_x + 1)],
                         dtype=torch.float32)  # .to(self.device)
        self.mask_x = torch.ones([self.size_x + 1, self.size_y], dtype=torch.float32).to(self.device)
        self.mask_y = torch.ones([self.size_x, self.size_y + 1], dtype=torch.float32).to(self.device)
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.obs_map[i,j]>0:
                    mask[i, j] = 1
        for i in range(self.size_x):
            for j in range(self.size_y):
                now = j * self.size_x + i
                left = j * (self.size_x + 1) + i
                right = j * (self.size_x + 1) + i + 1
                up = (j + 1) * self.size_x + i + (self.size_x + 1) * self.size_y
                down = j * self.size_x + i + (self.size_x + 1) * self.size_y
                GT[now][left] = 1
                GT[now][right] = -1
                GT[now][up] = -1
                GT[now][down] = 1

                if i > 0 and j > 0 and i < self.size_x - 1 and j < self.size_y - 1:
                    if mask[i - 1, j] != mask[i, j]: GT[now][left] = 0
                    if mask[i + 1, j] != mask[i, j]: GT[now][right] = 0
                    if mask[i, j - 1] != mask[i, j]: GT[now][down] = 0
                    if mask[i, j + 1] != mask[i, j]: GT[now][up] = 0

                if mask[i, j] == 1:
                    self.mask_x[i, j] = 0
                    self.mask_x[i + 1, j] = 0
                    self.mask_y[i, j] = 0
                    self.mask_y[i, j + 1] = 0

        self.GT = GT.to_sparse().to(self.device)

        self.G = GT.T.to(self.device)

        del (GT)

        alpha = 2.5e0 * (self.dx * self.dx)
        if self.use_sparse_FEM:

            I = torch.eye(self.size_x * self.size_y, self.size_x * self.size_y, dtype=torch.float32).to_sparse().to(
                self.device)
            self.L = (torch.matmul(self.GT, self.G) + alpha * I).to_sparse()

        else:
            I = torch.eye(self.size_x * self.size_y, self.size_x * self.size_y, dtype=torch.float32).to(self.device)
            self.L = (torch.matmul(self.GT, self.G) + alpha * I).inverse()

        self.G = self.G.to_sparse()

        del (mask)
        '''
        self.L=(torch.matmul(self.GT,self.G)+alpha*I).to_sparse()
        self.G=self.G.to_sparse()
        self.GT=self.GT.to_sparse()
        '''

    def P2G(self, pos, target):
        I = torch.zeros((pos.size(0), 1, self.size_x, self.size_y)).to(self.device)
        target_map = torch.zeros_like(I).to(self.device)
        target_map[:,0,int(self.aim[0]/self.dx),int(self.aim[1]/self.dx)]=1
        '''
        target_map=self.target_map.unsqueeze(0).unsqueeze(0).expand(
            [pos.size(0), 1, self.size_x, self.size_y]).to(self.device)
        target_map[target_map.clone()>=20]=20
        #target_map/=20
        '''
        '''
        obs_map=self.obs_map.unsqueeze(0).unsqueeze(0).expand(
            [pos.size(0), 1, self.size_x, self.size_y]).to(self.device)
        add_map=torch.cat((target_map,obs_map),1)
        '''
        target_map.requires_grad=True

        for i in range(pos.size(0)):
            px = pos[i, :self.n_robots]/self.scale
            py = pos[i, self.n_robots:]/self.scale

            idx = torch.floor(px / self.dx)
            idy = torch.floor(py / self.dx)
            alphax = (px - idx * self.dx)/self.dx
            alphay = (py - idy * self.dx)/self.dx

            I[i, 0, idx.long() + 1, idy.long() + 1] += 1*alphax * alphay
            I[i, 0, idx.long(), idy.long() + 1] += 1*(1 - alphax) * alphay
            I[i, 0, idx.long() + 1, idy.long()] += 1*alphax * (1 - alphay)
            I[i, 0, idx.long(), idy.long()] += 1*(1 - alphax) * (1 - alphay)

            # target[i,int(self.aim[0]/self.dx),int(self.aim[1]/self.dx)]=-1
        # I+=target

        input=torch.cat((I,target_map),1)
        '''
        for i in range(2):
            print(torch.std(input[:,i,:,:]))
        '''
        input[:,0,:,:]=input[:,0,:,:]/0.05#0.1
        input[:, 1, :, :] = input[:, 1, :, :] / 0.01#6
        #input[:, 2, :, :] = input[:, 2, :, :] / 0.4177
        #print(torch.mean(input))
        return input

    def apply(self,pos,action):
        t0 = time.time()
        k = action.size(0)
        x0 = action[:, ::5].unsqueeze(2)
        y0 = action[:, 1::5].unsqueeze(2)
        phix = (action[:, 2::5].unsqueeze(2) - 0.5) * 0.2
        phiy = (action[:, 3::5].unsqueeze(2) - 0.5) * 0.2
        alpha = (action[:, 4::5].unsqueeze(2)) * 0.0 + 1.0
        #alpha[alpha <= 1] = 1


        # print(torch.max(torch.abs(phix)))
        # print(action.grad)
        # alpha[alpha<=1]=1

        if self.use_kernel_loop == False:
            ux = pos[:,::2].unsqueeze(1)
            uy = pos[:,1::2].unsqueeze(1)

            r = torch.sqrt(torch.pow(x0 - ux, 2) + torch.pow(y0 - uy, 2) + self.eps)

            vel_x = (20 * torch.sum((0.1 * alpha + 1) * phix * torch.exp(-alpha * r),
                                         dim=1))
            vel_y = (20 * torch.sum((0.1 * alpha + 1) * phiy * torch.exp(-alpha * r),
                                         dim=1))
            v=1.0 * torch.cat((vel_x, vel_y), 1)
            v[v>2]=2
            v[v<-2]=-2
        return v
    def projection(self, action):
        # print(action.requires_grad)
        t0 = time.time()
        k = action.size(0)
        x0 = action[:, ::5].unsqueeze(2).unsqueeze(3)*2-0.5
        y0 = action[:, 1::5].unsqueeze(2).unsqueeze(3)*2-0.5
        phix = (action[:, 2::5].unsqueeze(2).unsqueeze(3) - 0.5) * 0.2
        phiy = (action[:, 3::5].unsqueeze(2).unsqueeze(3) - 0.5) * 0.2
        alpha = torch.square(action[:, 4::5].unsqueeze(2).unsqueeze(3)) * 19.0 + 0.8
        #alpha[alpha <= 1] = 1

        # print(phix)
        # print(action.grad)
        # alpha[alpha<=1]=1

        if self.use_kernel_loop == False:
            ux = self.grid_ux.unsqueeze(0).unsqueeze(0)
            uy = self.grid_uy.unsqueeze(0).unsqueeze(0)

            r = torch.sqrt(torch.pow(x0 - ux, 2) + torch.pow(y0 - uy, 2) + self.eps)

            velocity_x = (20 * torch.sum((0.1 * alpha + 1) * phix * torch.exp(-F.relu(alpha * r - 0.02) - 0.02),
                                         dim=1)) * self.mask_x

            vx = self.grid_vx.unsqueeze(0).unsqueeze(0)
            vy = self.grid_vy.unsqueeze(0).unsqueeze(0)

            r = torch.sqrt(torch.pow(vx - x0, 2) + torch.pow(vy - y0, 2) + self.eps)
            velocity_y = (20 * torch.sum((0.1 * alpha + 1) * phiy * torch.exp(-F.relu(alpha * r - 0.02) - 0.02),
                                         dim=1)) * self.mask_y
        else:
            ux = self.grid_ux.unsqueeze(0)
            uy = self.grid_uy.unsqueeze(0)
            vx = self.grid_vx.unsqueeze(0)
            vy = self.grid_vy.unsqueeze(0)
            velocity_x = torch.zeros(k, self.N, self.size_x, self.size_y)
            velocity_y = torch.zeros(k, self.N, self.size_x, self.size_y)
            for i in range(self.N):
                r = torch.sqrt(torch.pow(ux - x0[:, i, :, :], 2) + torch.pow(uy - y0[:, i, :, :], 2)+ self.eps)
                velocity_x += (torch.sum(
                    phix[:, i, :, :]  * torch.exp(-alpha[:, i, :, :] * r) * (uy - y0[:, i, :, :]) / r,
                    dim=1)) * self.mask_x
            for i in range(self.N):
                r = torch.sqrt(torch.pow(vx - x0[:, i, :, :], 2) + torch.pow(vy - y0[:, i, :, :], 2)+ self.eps)
                velocity_y += (torch.sum(
                    phiy[:, i, :, :]  * torch.exp(-alpha[:, i, :, :] * r) * (vx - x0[:, i, :, :]) / r,
                    dim=1)) * self.mask_y

        velocity = torch.cat(
            (torch.flatten(velocity_x.transpose(1, 2), 1), torch.flatten(velocity_y.transpose(1, 2), 1)),
            1)  # .unsqueeze(2)

        #velocity[velocity>3]=3
        #velocity[velocity<-3]=-3

        if self.use_sparse_FEM == False:
            velocity = (velocity).T
            v1 = torch.matmul(self.GT, velocity)
            v2 = torch.matmul(self.L, v1)
            v3 = torch.matmul(self.G, v2)
            velocity = velocity - v3
            velocity = (velocity.T).contiguous().view(k, -1)


        else:
            velocity = (velocity).T
            v1 = torch.matmul(self.GT, velocity)

            v2 = self.sparsesolve(self.L, v1)

            v3 = torch.matmul(self.G, v2)
            velocity = velocity - v3
            velocity = (velocity.T).contiguous().view(k, -1)

        return velocity  # .squeeze(2)

    def get_velocity(self, pos, velocity):
        # print(velocity.size())
        t0 = time.time()
        k = pos.size(0)

        # print((torch.sum(torch.abs(torch.matmul(self.GT,velocity)))))
        px = pos[:, :self.n_robots]
        py = pos[:, self.n_robots:]

        idx_x = torch.floor(px / self.dx)
        idx_y = torch.floor(py / self.dx)
        alpha_x = px / self.dx - idx_x
        alpha_y = py / self.dx - idx_y

        velocity_x = velocity[:, :(self.size_x + 1) * self.size_y]
        velocity_y = velocity[:, (self.size_x + 1) * self.size_y:]
        idx1 = (idx_y * (self.size_x + 1) + idx_x).long()

        vel_x = torch.gather(velocity_x, 1, idx1) * (1.0 - alpha_x) + torch.gather(velocity_x, 1, idx1 + 1) * alpha_x
        idx2 = (idx_y * self.size_x + idx_x).long()
        vel_y = torch.gather(velocity_y, 1, idx2) * (1.0 - alpha_y) + torch.gather(velocity_y, 1,
                                                                                   idx2 + self.size_x) * alpha_y
        '''
        # velocity boundary
        rr = torch.sqrt(vel_x * vel_x + vel_y * vel_y+ self.eps)

        energy = (torch.mean(rr, 1, keepdim=True) ) / 1.5

        vel_x = vel_x / energy
        vel_y = vel_y / energy
        '''
        rr = torch.sqrt(torch.square(vel_x) + torch.square(vel_y)+ 1e-2)
        #rr = F.relu(rr / 4.0 - 1.0) + 1.0

        return 1*torch.cat((vel_x /rr, vel_y /rr), 1)  # .squeeze(2)

        #return torch.cat((vel_x, vel_y), 1)  # .squeeze(2)

    def step(self, velocity):
        # action=action*0.5+0.5
        # velocity = velocity.squeeze(0)

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
            self.oldstate[i * 2:i * 2 + 2] = self.state[i * 2:i * 2 + 2]
        state_p = np.zeros([1, self.size_x, self.size_y])
        # self.BEM_solver()
        vx = velocity[:self.n_robots]
        vy = velocity[self.n_robots:]
        # vx = vx.cpu().detach().numpy()
        # vy = vy.cpu().detach().numpy()
        for it in range(1):
            pos = self.state.reshape([-1, 2])

            for i in range(self.n_robots):
                if i < self.n_robots / 2:
                    self.deltap[i] = [vx[i] * 0.1, vy[i] * 0.0]
                else:
                    self.deltap[i] = [vx[i] * 0.0, vy[i] * 0.1]

            # print(time.time()-t0)

            for i in range(self.n_robots):
                dx = vx[i]
                dy = vy[i]
                # print(dx,dy)
                lenn = math.sqrt(dx * dx + dy * dy)
                if lenn > 2.0:
                    dx *= 2.0 / lenn
                    dy *= 2.0 / lenn

                if self.state[i * 2] > 0.51 and self.state[i * 2] < 0.89 and self.state[i * 2 + 1] < 0.69 and \
                        self.state[i * 2 + 1] > 0.31:
                    self.t[i] -= 1
                    r = np.sqrt((pow(pos[i][0] - 0.7, 2) + pow(pos[i][1] - 0.5, 2)))
                    dx = 1.0 * (0.7 - pos[i][0]) / r
                    dy = 1.0 * (0.5 - pos[i][1]) / r
                    if r < 0.01 or self.t[i] <= 0:
                        dx = 0  # *=r/0.01
                        dy = 0  # *=r/0.01
                self.sim.setAgentPrefVelocity(self.agent[i], (dx, dy))
                self.sim.setAgentPosition(self.agent[i], (pos[i][0], pos[i][1]))
                self.deltap[i] = [dx, dy]

            self.sim.doNewtonStep(True)

            for i in range(self.n_robots):
                self.state[i * 2:i * 2 + 2] = self.sim.getAgentPosition(self.agent[i])
        for i in range(self.n_robots):
            self.oldvel[i * 2:i * 2 + 2] = self.vel[i * 2:i * 2 + 2]
            self.vel[i * 2:i * 2 + 2] = self.state[i * 2:i * 2 + 2] - self.oldstate[i * 2:i * 2 + 2]

        reward1, reward2 = get_reward(self.n_robots, self.aim, self.state, self.oldstate, self.angle, self.dis, self.dx)
        reward = reward1  # +reward2
        self.pos_reward += reward1
        self.vel_reward += reward2
        done = 0
        # state_p=P2G(self.state,state_p,self.n_robots,self.dx)
        # self.render(0)
        # print(time.time()-t0)
        '''
        a=np.append(self.x0,self.y0)
        b=np.append(self.omega,self.alpha)
        c=np.append(a,b)
        '''
        if self.cnt % 1 == 0:
            self.render()
        self.cnt += 1
        return self.state, reward, done, dict(reward=reward)

    def MBStep(self, xNew, render=True):
        X = xNew.detach().cpu().numpy()
        self.state = X.reshape(-1)
        if self.cnt % 1 == 0 and render:
            self.render()
        self.cnt += 1

    def MBLoss(self, xNew, x):

        loss = 0
        xNew=xNew/self.scale
        x=x/self.scale
        for i in range(xNew.size(0)):

            idx = torch.floor(xNew[i, :self.n_robots] / self.dx)
            idy = torch.floor(xNew[i, self.n_robots:] / self.dx)
            alphax = (xNew[i, :self.n_robots] - idx * self.dx)/self.dx
            alphay = (xNew[i, self.n_robots:] - idy * self.dx)/self.dx
            id = (idy * self.size_y + idx).long()
            distnew = self.dis[id + self.size_y + 1] * alphax * alphay + self.dis[id + self.size_y] * (
                    1.0 - alphax) * alphay \
                      + self.dis[id + 1] * alphax * (1.0 - alphay) + self.dis[id] * (
                              1.0 - alphax) * (1.0 - alphay)

            #if self.dis[id + self.size_y + 1].item()>15 or self.dis[id + self.size_y].item()>15 or self.dis[id + 1].item()>15 or self.dis[id].item()>15:
            #print(self.dis[id + self.size_y + 1], self.dis[id + self.size_y], self.dis[id + 1], self.dis[id])
            #loss+=torch.sum(distnew)

            idx = torch.floor(x[i, :self.n_robots] / self.dx)
            idy = torch.floor(x[i, self.n_robots:] / self.dx)
            alphax = (x[i, :self.n_robots] - idx * self.dx)/self.dx
            alphay = (x[i, self.n_robots:] - idy * self.dx)/self.dx
            id = (idy * self.size_y + idx).long()
            distold = self.dis[id + self.size_y + 1] * alphax * alphay + self.dis[id + self.size_y] * (
                    1.0 - alphax) * alphay \
                      + self.dis[id + 1] * alphax * (1.0 - alphay) + self.dis[id] * (
                              1.0 - alphax) * (1.0 - alphay)
            loss += torch.sum(torch.square(distnew) - torch.square(distold))
            #loss += torch.sum(distnew - distold)
            #loss+=torch.sum(xNew-x)

        return 0.1*loss/xNew.size(0)
        '''
        xNew=xNew/self.scale
        #return torch.sum(torch.square(xNew[::2]-self.aim[0])+torch.square(xNew[1::2]-self.aim[1]))
        return torch.sum(torch.square(xNew[:,:self.n_robots] - self.aim[0]) + torch.square(xNew[:,self.n_robots:] - self.aim[1]))
        '''
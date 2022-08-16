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


class CoverEnvs():
    def __init__(self, batch_size,gui, sim, multisim,use_kernel_loop, use_sparse_FEM):

        self.batch_size=batch_size
        self.sim = sim
        self.multisim=multisim

        self.gui = gui
        self.use_kernel_loop = use_kernel_loop
        self.use_sparse_FEM = use_sparse_FEM
        self.current_obs=[]
        self.viewer=None

        self.N = 25  # kernel number
        self.radius = 0.008  # robot radius
        self.n_robots = 100  # robot number
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
        self.gt = torch.zeros([self.size_x, self.size_y])

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
        current_obs=[]
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
        #for i in range(self.N):
        #    self.viewer.add_waypoint((self.x0[i]*self.wind_size[0],self.y0[i]*self.wind_size[0]),self.radius*self.wind_size[0])
        #for p in self.goal_positions:
        #self.viewer.add_goal((self.aim[0]*self.wind_size[0], self.aim[1]*self.wind_size[0]), 0.016*self.wind_size[0])
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
        #for i in range(self.N):
        #    self.viewer.waypoint_pos_array[i]=(self.x0[i]*self.wind_size[0],self.y0[i]*self.wind_size[0])
        #self.viewer.goal_pos_array[0] = (self.aim[0]*self.wind_size[0], self.aim[1]*self.wind_size[0])
        self.viewer.render()
    def reset_init_agent(self):
        x=0.105
        idx=0
        while x<0.9:
            y = 0.205
            while y<0.4:
                self.init_state.append([x,y])
                y+=0.04
                #print(x,y)
            x+=0.04
        #self.reset_agent()
        self.reset_aim()
    def reset_agent(self):
        agent_no=self.init_state#random.sample(self.init_state, self.n_robots)
        for i in range(self.n_robots):
            self.state[i] = agent_no[i][0]
            self.state[i + self.n_robots] = agent_no[i][1]
        return self.state*self.scale
    def reset_aim(self):
        state=np.zeros([1,self.n_robots*2])
        x=0.305
        idx=0
        while x<0.7:
            y=0.305
            while y<0.7:
                state[0,idx]=x*self.scale
                state[0,idx+self.n_robots]=y*self.scale
                idx+=1

                y+=0.04
            x+=0.04
        self.gt=self.P2GForLoss(torch.from_numpy(state).to(self.device))
    def reset(self):
        self.reset_init_agent()
        self.dis = dijkstra(self.dx, find_grid_index(self.aim, self.dx), self.obs_map).to(self.device)
        self.dis[self.dis>100]=20

        self.get_target_map()

        if self.viewer is not None:
            self.viewer.reset_array()
            self.reset_viewer()

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

    def P2G(self, pos):
        I = torch.zeros((pos.size(0), 1, self.size_x, self.size_y)).to(self.device)

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

        I=I/0.05

        return I
    def P2GForLoss(self, pos,sz=100):
        dx=1.0/sz

        I = torch.zeros((pos.size(0), 1, sz, sz)).to(self.device)

        for i in range(pos.size(0)):
            px = pos[i, :self.n_robots]/self.scale
            py = pos[i, self.n_robots:]/self.scale

            idx = torch.floor(px / dx)
            idy = torch.floor(py / dx)
            alphax = (px - idx * dx)/dx
            alphay = (py - idy * dx)/dx

            I[i, 0, idx.long() + 1, idy.long() + 1] += 1*alphax * alphay
            I[i, 0, idx.long(), idy.long() + 1] += 1*(1 - alphax) * alphay
            I[i, 0, idx.long() + 1, idy.long()] += 1*alphax * (1 - alphay)
            I[i, 0, idx.long(), idy.long()] += 1*(1 - alphax) * (1 - alphay)
        #print(I)
        I=I/0.05

        return I

    def projection(self, action):
        # print(action.requires_grad)
        t0 = time.time()
        k = action.size(0)
        x0 = action[:, ::5].unsqueeze(2).unsqueeze(3) * 1.2 - 0.1
        y0 = action[:, 1::5].unsqueeze(2).unsqueeze(3) * 1.2 - 0.1
        phix = (action[:, 2::5].unsqueeze(2).unsqueeze(3) - 0.5) * 0.2
        phiy = (action[:, 3::5].unsqueeze(2).unsqueeze(3) - 0.5) * 0.2
        alpha = (action[:, 4::5].unsqueeze(2).unsqueeze(3)) * 15.5 + 0.5
        #alpha[alpha <= 1] = 1

        # print(phix)
        # print(action.grad)
        # alpha[alpha<=1]=1

        if self.use_kernel_loop == False:
            ux = self.grid_ux.unsqueeze(0).unsqueeze(0)
            uy = self.grid_uy.unsqueeze(0).unsqueeze(0)

            r = torch.sqrt(torch.pow(x0 - ux, 2) + torch.pow(y0 - uy, 2) + self.eps)

            velocity_x = (4 * torch.sum((0.1 * alpha + 1) * phix * torch.exp(-alpha * r ),
                                        dim=1)) * self.mask_x

            vx = self.grid_vx.unsqueeze(0).unsqueeze(0)
            vy = self.grid_vy.unsqueeze(0).unsqueeze(0)

            r = torch.sqrt(torch.pow(vx - x0, 2) + torch.pow(vy - y0, 2) + self.eps)
            velocity_y = (4 * torch.sum((0.1 * alpha + 1) * phiy * torch.exp(-alpha * r ),
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

    def get_velocity(self, pos, velocity,is_norm=True):
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

        rr = torch.sqrt(torch.square(vel_x) + torch.square(vel_y))

        rr = F.relu(rr / 1.0 - 1.0) + 1.0
        '''
        vel_x/=rr
        vel_y/=rr
        
        rr = torch.sqrt(torch.square(vel_x) + torch.square(vel_y) + self.eps)
        vel_x[rr<0.1]=0
        vel_y[rr < 0.1] = 0
        '''
        return torch.cat((vel_x/rr, vel_y/rr), 1)  # .squeeze(2)


    def MBStep(self, xNew, render=True):
        X = xNew.detach().cpu().numpy()
        self.state = X.reshape(-1)
        if self.cnt % 1 == 0 and render:
            self.render()
        self.cnt += 1

    def ShapeLoss(self,xNew,x):
        '''
        loss=torch.nn.MSELoss()
        #print(xNew,self.gt)
        return loss(xNew,self.gt)
        '''
        I=self.P2GForLoss(xNew)

        loss=-1e-5*(torch.square(torch.sum(I[:,:,30:68,30:68])-torch.sum(self.gt[:,:,30:68,30:68])))/xNew.size(0)
        return loss

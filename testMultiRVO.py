import sys
sys.path.append('../kernel-based-navigation-build')
import pyRVO as pyrvo
import numpy as np

#you can change this to be very large
maxVelocity=1
batchSize=5

#add obstacle
rvo=pyrvo.MultiRVOSimulator(batchSize,2)
for off in [np.array([-70.,-70.]),np.array([30.,-70.]),np.array([30.,30.]),np.array([-70.,30.])]:
    v=[off+np.array([ 0., 0.]),
       off+np.array([40., 0.]),
       off+np.array([40.,40.]),
       off+np.array([ 0.,40.]),]
    id=rvo.addObstacle(v)
    print('Obstacle %d:'%id)
    for v in rvo.getObstacle(id):
        print(v.T,end='')
    print('')
    
#add agent
for x in range(-120,-80,10):
    for y in range(-120,-80,10):
        pos,vel,tar=[],[],[]
        for i in range(batchSize):
            pos.append(np.array([x,y],dtype=float))
            vel.append(np.array([0.,0.]))
            tar.append(-pos[-1])
        id=rvo.addAgent(pos,vel)
        rvo.setAgentTarget(id,tar,maxVelocity)
for x in range(-120,-80,10):
    for y in range(80,120,10):
        pos,vel,tar=[],[],[]
        for i in range(batchSize):
            pos.append(np.array([x,y],dtype=float))
            vel.append(np.array([0.,0.]))
            tar.append(-pos[-1])
        id=rvo.addAgent(pos,vel)
        rvo.setAgentTarget(id,tar,maxVelocity)
for x in range(80,120,10):
    for y in range(-120,-80,10):
        pos,vel,tar=[],[],[]
        for i in range(batchSize):
            pos.append(np.array([x,y],dtype=float))
            vel.append(np.array([0.,0.]))
            tar.append(-pos[-1])
        id=rvo.addAgent(pos,vel)
        rvo.setAgentTarget(id,tar,maxVelocity)
for x in range(80,120,10):
    for y in range(80,120,10):
        pos,vel,tar=[],[],[]
        for i in range(batchSize):
            pos.append(np.array([x,y],dtype=float))
            vel.append(np.array([0.,0.]))
            tar.append(-pos[-1])
        id=rvo.addAgent(pos,vel)
        rvo.setAgentTarget(id,tar,maxVelocity)
        
#simulate
while True:
    rvo.updateAgentTargets()
    rvo.optimize(False,True)
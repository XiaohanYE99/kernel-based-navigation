import sys
import pyRVO as pyrvo
import numpy as np

#you can change this to be very large
maxVelocity=1

#add obstacle
rvo=pyrvo.RVOSimulator(2)
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
rad=2
for x in range(-120,-80,10):
    for y in range(-120,-80,10):
        id=rvo.addAgent(np.array([x,y],dtype=float),np.array([0.,0.]),rad)
        rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxVelocity)
rad=1
for x in range(-120,-80,10):
    for y in range(80,120,10):
        id=rvo.addAgent(np.array([x,y],dtype=float),np.array([0.,0.]),rad)
        rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxVelocity)
rad=0.5
for x in range(80,120,10):
    for y in range(-120,-80,10):
        id=rvo.addAgent(np.array([x,y],dtype=float),np.array([0.,0.]),rad)
        rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxVelocity)
rad=2
for x in range(80,120,10):
    for y in range(80,120,10):
        id=rvo.addAgent(np.array([x,y],dtype=float),np.array([0.,0.]),rad)
        rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxVelocity)
        
#print
print('agent positions=',rvo.getAgentPositions())
print('agent velocities=',rvo.getAgentVelocities())

#simulate
pyrvo.drawRVOApp(100,rvo)

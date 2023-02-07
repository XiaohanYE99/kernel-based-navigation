import sys,random
import pyRVO as pyrvo
import numpy as np

scale=100

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
for i in range(100):
    id=rvo.addAgent(np.array([random.randrange(-scale,scale),
                              random.randrange(-scale,scale)],dtype=float),np.array([0.,0.]),1.0)
    
#debug
pos=np.zeros((100*2,),dtype=float)
for i in range(200):
    pos[i]=random.randrange(-scale,scale)

#single coverage energy
C=pyrvo.CoverageEnergy(rvo,50,True)
loss=C.loss(pos)
grad=C.grad()

#debug FD
pyrvo.CoverageEnergy(rvo,50,True).debugCoverage(scale)
pyrvo.CoverageEnergy(rvo,50,False).debugCoverage(scale)
rvo.debugNeighbor(scale)
rvo.debugEnergy(scale)

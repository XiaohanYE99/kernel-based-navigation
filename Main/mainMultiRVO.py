import random
from utils import *

def setup_multi_RVO():
    #you can change this to be very large
    maxVelocity=1
    batchSize=15

    #add obstacle
    rvo=pyrvo.MultiRVOSimulator(batchSize,1,1e-4,1,1,10,False,True,"NEWTON")
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
            pos,vel,rad,tar=[],[],[],[]
            for i in range(batchSize):
                pos.append(np.array([x+random.randrange(-3,3),y+random.randrange(-3,3)],dtype=float))
                vel.append(np.array([0.,0.]))
                tar.append(-pos[-1])
                rad.append(2)
            id=rvo.addAgent(pos,vel,rad)
            rvo.setAgentTarget(id,tar,maxVelocity)
    for x in range(-120,-80,10):
        for y in range(80,120,10):
            pos,vel,rad,tar=[],[],[],[]
            for i in range(batchSize):
                pos.append(np.array([x+random.randrange(-3,3),y+random.randrange(-3,3)],dtype=float))
                vel.append(np.array([0.,0.]))
                tar.append(-pos[-1])
                rad.append(1)
            id=rvo.addAgent(pos,vel,rad)
            rvo.setAgentTarget(id,tar,maxVelocity)
    for x in range(80,120,10):
        for y in range(-120,-80,10):
            pos,vel,rad,tar=[],[],[],[]
            for i in range(batchSize):
                pos.append(np.array([x+random.randrange(-3,3),y+random.randrange(-3,3)],dtype=float))
                vel.append(np.array([0.,0.]))
                tar.append(-pos[-1])
                rad.append(0.5)
            id=rvo.addAgent(pos,vel,rad)
            rvo.setAgentTarget(id,tar,maxVelocity)
    for x in range(80,120,10):
        for y in range(80,120,10):
            pos,vel,rad,tar=[],[],[],[]
            for i in range(batchSize):
                pos.append(np.array([x+random.randrange(-3,3),y+random.randrange(-3,3)],dtype=float))
                vel.append(np.array([0.,0.]))
                tar.append(-pos[-1])
                rad.append(2)
            id=rvo.addAgent(pos,vel,rad)
            rvo.setAgentTarget(id,tar,maxVelocity)
    return rvo

if __name__=='__main__':
    id = 0
    sim = False
    css = {}
    css[0] = [1, 0, 0]
    css[1] = [0, 1, 0]
    css[2] = [0, 0, 1]
    css[3] = [1, 0, 1]
    rvo = setup_multi_RVO()
    drawer,shapes,export,capturer = setup_visualizer(rvo, 100, css, 0)
    def key(wnd,key,scan,action,mods,captured):
        global sim,id
        if captured:
            return
        if key == vis.GLFW_KEY_R and action == vis.GLFW_PRESS:
            sim = not sim
        elif key == vis.GLFW_KEY_D and action == vis.GLFW_PRESS:
            id=(id+1)%rvo.getBatchSize()
            draw_RVO(rvo, shapes, css, id)
        elif key == vis.GLFW_KEY_A and action == vis.GLFW_PRESS:
            id=(id+rvo.getBatchSize()-1)%rvo.getBatchSize()
            draw_RVO(rvo, shapes, css, id)
    def frame(sceneRoot):
        global sim,rvo,shapes,id
        if sim:
            rvo.updateAgentTargets()
            rvo.optimize(False, False)
            draw_RVO(rvo, shapes, css, id)
    #initiate main loop
    drawer.setKeyFunc(key)
    drawer.setFrameFunc(frame)
    drawer.mainLoop()
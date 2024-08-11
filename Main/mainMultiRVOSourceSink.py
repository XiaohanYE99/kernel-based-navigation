import pyRVO

from utils import *
def setup_multi_RVO_SourceSink():
    # you can change this to be very large
    maxVelocity = 1
    batchSize=15
    noise=5.

    # add obstacle
    rvo = pyrvo.MultiRVOSimulator(batchSize, 1, 1e-4, 1, 1, 1000, False, True, "NEWTON")
    rvo.clearAgent()
    rvo.clearObstacle()
    rvo.clearSourceSink()
    rvo.clearVisibility()

    # add source sink
    rvo.setupSourceSink(1,10,True)
    rvo.addSourceSink(np.array([120.,120.]),np.array([-120.,-120.]),np.array([-130.,-130.]),np.array([-110.,-110.]),4,noise)
    rvo.addSourceSink(np.array([-120.,-120.]),np.array([120.,120.]),np.array([110.,110.]),np.array([130.,130.]),5,noise)
    rvo.addSourceSink(np.array([-120.,120.]),np.array([120.,-120.]),np.array([110.,-130.]),np.array([130.,-110.]),4,noise)
    rvo.addSourceSink(np.array([120.,-120.]),np.array([-120.,120.]),np.array([-130.,110.]),np.array([-110.,130.]),5,noise)
    rvo.addObstacle([np.array([-10.,-10.]), np.array([10.,-10.]), np.array([10.,10.]), np.array([-10.,10.])])
    rvo.buildVisibility()
    return rvo

if __name__=='__main__':
    id = 0
    sim = False
    css = {}
    css[0] = [1, 0, 0]
    css[1] = [0, 1, 0]
    css[2] = [0, 0, 1]
    css[3] = [1, 0, 1]
    rvo = setup_multi_RVO_SourceSink()
    drawer,shapes = setup_visualizer(rvo, 100, css)
    def key(wnd,key,scan,action,mods,captured):
        global sim,id
        if captured:
            return
        if key == vis.GLFW_KEY_R and action == vis.GLFW_PRESS:
            sim = not sim
        elif key == vis.GLFW_KEY_D and action == vis.GLFW_PRESS:
            id=(id+1)%rvo.getBatchSize()
            draw_RVO(rvo.getSubSimulator(id), shapes, css)
        elif key == vis.GLFW_KEY_A and action == vis.GLFW_PRESS:
            id=(id+rvo.getBatchSize()-1)%rvo.getBatchSize()
            draw_RVO(rvo.getSubSimulator(id), shapes, css)
    def frame(sceneRoot):
        global sim,rvo,shapes,id
        if sim:
            rvo.updateAgentTargets()
            rvo.optimize(False, False)
            draw_RVO(rvo.getSubSimulator(id), shapes, css)
    #initiate main loop
    drawer.setKeyFunc(key)
    drawer.setFrameFunc(frame)
    drawer.mainLoop()
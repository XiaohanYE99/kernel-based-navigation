import pyRVO

from utils import *
def setup_RVO_SourceSink():
    # you can change this to be very large
    maxVelocity = 1

    # add obstacle
    rvo = pyrvo.RVOSimulator(1, 1e-4, 1, 1, 1000, False, True, "NEWTON")
    rvo.addObstacle([np.array([-10.,-10.]), np.array([10.,-10.]), np.array([10.,10.]), np.array([-10.,10.])])
    rvo.buildVisibility()

    # add source sink
    ss = pyRVO.SourceSink(1,10,True)
    ss.addSourceSink(np.array([120.,120.]),np.array([-120.,-120.]),pyrvo.BBox(np.array([-130.,-130.]),np.array([-110.,-110.])),4)
    ss.addSourceSink(np.array([-120.,-120.]),np.array([120.,120.]),pyrvo.BBox(np.array([110.,110.]),np.array([130.,130.])),5)
    ss.addSourceSink(np.array([-120.,120.]),np.array([120.,-120.]),pyrvo.BBox(np.array([110.,-130.]),np.array([130.,-110.])),4)
    ss.addSourceSink(np.array([120.,-120.]),np.array([-120.,120.]),pyrvo.BBox(np.array([-130.,110.]),np.array([-110.,130.])),5)
    return rvo,ss

if __name__=='__main__':
    frameId = 0
    sim = False
    css = {}
    css[0] = [1, 0, 0]
    css[1] = [0, 1, 0]
    css[2] = [0, 0, 1]
    css[3] = [1, 0, 1]
    rvo,ss = setup_RVO_SourceSink()
    drawer,shapes = setup_visualizer(rvo, 100, css)
    def key(wnd,key,scan,action,mods,captured):
        global sim,frameId
        if captured:
            return
        if key == vis.GLFW_KEY_R and action == vis.GLFW_PRESS:
            sim = not sim
    def frame(sceneRoot):
        global sim,rvo,shapes,frameId
        if sim:
            rvo.updateAgentTargets()
            rvo.optimize(False, False)
            ss.recordAgents(rvo)
            ss.addAgents(frameId,rvo,1.e-4)
            ss.removeAgents(rvo)
            frameId += 1
            draw_RVO(rvo, shapes, css)
    #initiate main loop
    drawer.setKeyFunc(key)
    drawer.setFrameFunc(frame)
    drawer.mainLoop()
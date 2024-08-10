import sys
import pyRVO as pyrvo
import pyTinyVisualizer as vis
import numpy as np

COLOR_AGT= [200/255.,143/255., 29/255.]
COLOR_OBS= [000/255.,000/255.,000/255.]
COLOR_VEL= [120/255.,000/255.,000/255.]
COLOR_VIS= [000/255.,255/255.,000/255.]
css = {}
css[0]=[1,0,0]
css[1]=[0,1,0]
css[2]=[0,0,1]
css[3]=[1,0,1]

def draw_RVO_obstacles(rvo, shapes):
    obss=vis.CompositeShape()
    for i in range(rvo.getNrObstacle()):
        pos=rvo.getObstacle(i)
        obs=vis.MeshShape()
        for j in range(len(pos)-2):
            obs.addIndexSingle(obs.nrVertex()+0)
            obs.addIndexSingle(obs.nrVertex()+1)
            obs.addIndexSingle(obs.nrVertex()+2)
            obs.addVertex([pos[0  ][0],pos[0  ][1],0],[0,0])
            obs.addVertex([pos[j+1][0],pos[j+1][1],0],[0,0])
            obs.addVertex([pos[j+2][0],pos[j+2][1],0],[0,0])
        obs.setMode(vis.GL_TRIANGLES)
        obs.setColorDiffuse(vis.GL_TRIANGLES,COLOR_OBS[0],COLOR_OBS[1],COLOR_OBS[2])
        obss.addShape(obs)
    shapes.addShape(obss)

def draw_RVO(rvo, shapes):
    if shapes is None:
        shapes=vis.CompositeShape()
        draw_RVO_obstacles(rvo, shapes)
    #more children
    while shapes.numChildren() < rvo.getNrAgent()+1:
        agent = vis.Bullet3DShape()
        circle = vis.makeCircle(vis.GL_TRIANGLE_FAN,True,[0,0],1)
        circle.setColorDiffuse(vis.GL_TRIANGLE_FAN,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2])
        agent.addShape(circle)
        shapes.addShape(agent)
    #less children
    while shapes.numChildren() > rvo.getNrAgent()+1:
        shapes.addShape(shapes.getChild(shapes.numChildren()-1))
    #update translation
    for i in range(rvo.getNrAgent()):
        t=np.identity(4)
        t[0,0]*=rvo.getAgentRadius(i)
        t[1,1]*=rvo.getAgentRadius(i)
        t[0,3]=rvo.getAgentPosition(i)[0]
        t[1,3]=rvo.getAgentPosition(i)[1]
        #set agent color
        shape=shapes.getChild(i+1)
        sid=pyrvo.SourceSink.extractSourceId(rvo.getAgentId(i))
        if sid in css:
            shape.setColorDiffuse(vis.GL_TRIANGLE_FAN,css[sid][0],css[sid][1],css[sid][2])
        shape.setColorDiffuse(vis.GL_TRIANGLE_FAN,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2])
        shape.setLocalTransform(t)
    return shapes

def setup_visualizer(rvo, ext):
    drawer = vis.Drawer([])
    export = vis.CameraExportPlugin(vis.GLFW_KEY_2,vis.GLFW_KEY_3,"camera.dat")
    capturer = vis.CaptureGIFPlugin(vis.GLFW_KEY_1,"record.gif",drawer.FPS(),True)
    drawer.addPlugin(export)
    drawer.addPlugin(capturer)
    shapes = draw_RVO(rvo, None)
    drawer.addShape(shapes)
    drawer.addCamera2D(ext)
    drawer.clearLight()
    return drawer, shapes

def setup_RVO():
    # you can change this to be very large
    maxVelocity = 1

    # add obstacle
    rvo = pyrvo.RVOSimulator(1, 1e-4, 1, 1, 1000, False, True, "NEWTON")
    for off in [np.array([-70., -70.]), np.array([30., -70.]), np.array([30., 30.]), np.array([-70., 30.])]:
        v = [off + np.array([0., 0.]),
             off + np.array([40., 0.]),
             off + np.array([40., 40.]),
             off + np.array([0., 40.]), ]
        id = rvo.addObstacle(v)
        print('Obstacle %d:' % id)
        for v in rvo.getObstacle(id):
            print(v.T, end='')
        print('')

    # add agent
    rad = 2
    for x in range(-120, -80, 10):
        for y in range(-120, -80, 10):
            id = rvo.addAgent(np.array([x, y], dtype=float), np.array([0., 0.]), rad, -1)
            rvo.setAgentTarget(id, -rvo.getAgentPosition(id), maxVelocity)
    rad = 1
    for x in range(-120, -80, 10):
        for y in range(80, 120, 10):
            id = rvo.addAgent(np.array([x, y], dtype=float), np.array([0., 0.]), rad, -1)
            rvo.setAgentTarget(id, -rvo.getAgentPosition(id), maxVelocity)
    rad = 0.5
    for x in range(80, 120, 10):
        for y in range(-120, -80, 10):
            id = rvo.addAgent(np.array([x, y], dtype=float), np.array([0., 0.]), rad, -1)
            rvo.setAgentTarget(id, -rvo.getAgentPosition(id), maxVelocity)
    rad = 2
    for x in range(80, 120, 10):
        for y in range(80, 120, 10):
            id = rvo.addAgent(np.array([x, y], dtype=float), np.array([0., 0.]), rad, -1)
            rvo.setAgentTarget(id, -rvo.getAgentPosition(id), maxVelocity)

    # print
    print('agent positions=', rvo.getAgentPositions())
    print('agent velocities=', rvo.getAgentVelocities())
    return rvo

if __name__=='__main__':
    sim = False
    rvo = setup_RVO()
    drawer,shapes = setup_visualizer(rvo, 100)
    def key(wnd,key,scan,action,mods,captured):
        global sim
        if captured:
            return
        if key == vis.GLFW_KEY_R and action == vis.GLFW_PRESS:
            sim = not sim
    def frame(sceneRoot):
        global sim,rvo,shapes
        if sim:
            rvo.updateAgentTargets()
            rvo.optimize(False, False)
            draw_RVO(rvo, shapes)
    #initiate main loop
    drawer.setKeyFunc(key)
    drawer.setFrameFunc(frame)
    drawer.mainLoop()
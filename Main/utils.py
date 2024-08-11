import sys
import pyRVO as pyrvo
import pyTinyVisualizer as vis
import numpy as np

COLOR_AGT= [200/255.,143/255., 29/255.]
COLOR_OBS= [000/255.,000/255.,000/255.]
COLOR_VEL= [120/255.,000/255.,000/255.]
COLOR_VIS= [000/255.,255/255.,000/255.]

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

def draw_RVO(rvo, shapes, css):
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

def setup_visualizer(rvo, ext, css):
    if isinstance(rvo,pyrvo.MultiRVOSimulator):
        rvo=rvo.getSubSimulator(0)
    drawer = vis.Drawer([])
    export = vis.CameraExportPlugin(vis.GLFW_KEY_2,vis.GLFW_KEY_3,"camera.dat")
    capturer = vis.CaptureGIFPlugin(vis.GLFW_KEY_1,"record.gif",drawer.FPS(),True)
    drawer.addPlugin(export)
    drawer.addPlugin(capturer)
    shapes = draw_RVO(rvo, None, css)
    drawer.addShape(shapes)
    drawer.addCamera2D(ext)
    drawer.clearLight()
    return drawer, shapes

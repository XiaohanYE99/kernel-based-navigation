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

def get_RVO_radius(rvo, batchId):
    if isinstance(rvo,pyrvo.MultiRVOSimulator):
        return rvo.getAllAgentRadius(batchId)
    else: return rvo.getAgentRadius()

def get_RVO_positions(rvo, batchId):
    if isinstance(rvo,pyrvo.MultiRVOSimulator):
        return np.array(rvo.getAllAgentPositions(batchId))
    else: return np.array(rvo.getAgentPositions())

def get_RVO_agentIds(rvo, batchId):
    if isinstance(rvo,pyrvo.MultiRVOSimulator):
        return rvo.getAllAgentIds(batchId)
    else: return rvo.getAgentId()

def draw_RVO(rvo, shapes, css, batchId=None):
    if shapes is None:
        shapes=vis.CompositeShape()
        draw_RVO_obstacles(rvo, shapes)
    #get information
    rads = get_RVO_radius(rvo, batchId)
    poss = get_RVO_positions(rvo, batchId)
    sids = get_RVO_agentIds(rvo, batchId)
    #more children
    while shapes.numChildren() < poss.shape[1]+1:
        agent = vis.Bullet3DShape()
        circle = vis.makeCircle(16,True,[0,0],1)
        circle.setColorDiffuse(vis.GL_TRIANGLE_FAN,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2])
        agent.addShape(circle)
        shapes.addShape(agent)
    #less children
    while shapes.numChildren() > poss.shape[1]+1:
        shapes.removeChild(shapes.getChild(shapes.numChildren()-1))
    #update translation
    for i in range(poss.shape[1]):
        t=np.identity(4)
        t[0,0]*=rads[i]
        t[1,1]*=rads[i]
        t[0,3]=poss[0,i]
        t[1,3]=poss[1,i]
        #set agent color
        shape=shapes.getChild(i+1)
        sid=pyrvo.SourceSink.extractSourceId(sids[i])
        if sid in css:
            shape.setColorDiffuse(vis.GL_TRIANGLE_FAN,css[sid][0],css[sid][1],css[sid][2])
        else: shape.setColorDiffuse(vis.GL_TRIANGLE_FAN,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2])
        shape.setLocalTransform(t)
    return shapes

def setup_visualizer(rvo, ext, css, batchId=None):
    drawer = vis.Drawer([])
    export = vis.CameraExportPlugin(vis.GLFW_KEY_2,vis.GLFW_KEY_3,"camera.dat")
    capturer = vis.CaptureGIFPlugin(vis.GLFW_KEY_1,"record.gif",drawer.FPS(),True)
    drawer.addPlugin(export)
    drawer.addPlugin(capturer)
    shapes = draw_RVO(rvo, None, css, batchId)
    drawer.addShape(shapes)
    drawer.addCamera2D(ext)
    drawer.clearLight()
    return drawer, shapes

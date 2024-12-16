from utils import *
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
    offlineRecording = True
    sim = offlineRecording
    css = {}
    css[0] = [1, 0, 0]
    css[1] = [0, 1, 0]
    css[2] = [0, 0, 1]
    css[3] = [1, 0, 1]
    rvo = setup_RVO()
    drawer,shapes,export,capturer = setup_visualizer(rvo, 100, css)
    def key(wnd,key,scan,action,mods,captured):
        global sim,capturer
        if captured:
            return
        if key == vis.GLFW_KEY_R and action == vis.GLFW_PRESS:
            sim = not sim
        if key == vis.GLFW_KEY_T and action == vis.GLFW_PRESS:
            take_screenshot(capturer,'screenshot.png')
    def frame(sceneRoot):
        global sim,rvo,shapes
        if sim:
            rvo.updateAgentTargets()
            rvo.optimize(False, False)
            draw_RVO(rvo, shapes, css)
    #initiate main loop
    drawer.setKeyFunc(key)
    drawer.setFrameFunc(frame)
    if offlineRecording:
        record_video(drawer, capturer, 100, 'record.mp4')
    else:
        drawer.mainLoop()
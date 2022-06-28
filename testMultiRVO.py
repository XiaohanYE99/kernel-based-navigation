import rvo2,random

n_agent = 500
batch_size = 1024
print("Start testing multi-RVO")
print("============================================================================================")
sim = rvo2.PyRVOMultiSimulator(batch_size, 3 / 600., 0.03, 5, 0.04, 0.04, 0.01, 2)
sim.setTimeStep(0.25)
sim.setAgentDefaults(15.0, 10, 5.0, 5.0, 2.0, 2.0)
    
#obstacle
for a,b in [(2,2),(4,4),(2,4),(4,2)]:
    obstacle=[(a,b),(a+1,b),(a+1,b+1),(a,b+1)]
    sim.addObstacle(obstacle)
print('#Obstacle-Vertex=%d'%sim.getNumObstacleVertices())
sim.processObstacles()

#agent
for i in range(n_agent):
    sim.addAgent([(random.uniform(-1,1),random.uniform(-1,1)) for j in range(batch_size)])
print([(random.uniform(-1,1),random.uniform(-1,1)) for j in range(batch_size)])
print("#Agents=%d"%sim.getNumAgents())

#step
sim.doStep()

#clear
sim.clearObstacle()
print("============================================================================================")
print("End testing multi-RVO")
#include "Visualizer.h"
#include <TinyVisualizer/Drawer.h>
#include <TinyVisualizer/Camera2D.h>
#include <TinyVisualizer/MakeMesh.h>
#include <TinyVisualizer/MeshShape.h>
#include <TinyVisualizer/Bullet3DShape.h>
#include <TinyVisualizer/CompositeShape.h>
#include <TinyVisualizer/CaptureGIFPlugin.h>
#include <TinyVisualizer/CameraExportPlugin.h>

namespace RVO {
std::shared_ptr<DRAWER::CompositeShape> drawRVO(const RVOSimulator& sim,std::shared_ptr<DRAWER::CompositeShape> shapesInput) {
  using namespace DRAWER;
  std::shared_ptr<CompositeShape> shapes=shapesInput?shapesInput:std::shared_ptr<CompositeShape>(new CompositeShape);
  if(!shapesInput) {
    std::shared_ptr<MeshShape> circle=makeCircle(16,true,Eigen::Matrix<GLfloat,2,1>::Zero(),(GLfloat)sim.getRadius());
    circle->setColor(GL_TRIANGLES,0,0,0);
    for(int i=0; i<sim.getNrAgent(); i++) {
      std::shared_ptr<Bullet3DShape> agent(new Bullet3DShape);
      agent->addShape(circle);
      shapes->addShape(agent);
    }
    for(int i=0; i<sim.getNrObstacle(); i++) {
      std::vector<RVOSimulator::Vec2T> pos=sim.getObstacle(i);
      std::shared_ptr<MeshShape> obs(new MeshShape);
      //must be convex
      for(int j=0; j<(int)pos.size()-2; j++) {
        obs->addIndexSingle(obs->nrVertex()+0);
        obs->addIndexSingle(obs->nrVertex()+1);
        obs->addIndexSingle(obs->nrVertex()+2);
        obs->addVertex(Eigen::Matrix<GLfloat,3,1>((GLfloat)pos[0  ][0],(GLfloat)pos[0  ][1],0));
        obs->addVertex(Eigen::Matrix<GLfloat,3,1>((GLfloat)pos[j+1][0],(GLfloat)pos[j+1][1],0));
        obs->addVertex(Eigen::Matrix<GLfloat,3,1>((GLfloat)pos[j+2][0],(GLfloat)pos[j+2][1],0));
      }
      obs->setMode(GL_TRIANGLES);
      obs->setColor(GL_TRIANGLES,0,0,0);
      shapes->addShape(obs);
    }
  }
  for(int i=0; i<sim.getNrAgent(); i++)
    std::dynamic_pointer_cast<Bullet3DShape>(shapes->getChild(i))->
    setLocalTranslate(Eigen::Matrix<GLfloat,3,1>((GLfloat)sim.getAgentPosition(i)[0],(GLfloat)sim.getAgentPosition(i)[1],0));
  return shapes;
}
void drawRVOApp(int argc,char** argv,GLfloat ext,const RVOSimulator& sim,std::function<void()> frm) {
  using namespace DRAWER;
  Drawer drawer(argc,argv);
  drawer.addPlugin(std::shared_ptr<Plugin>(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat")));
  drawer.addPlugin(std::shared_ptr<Plugin>(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",drawer.FPS())));
  std::shared_ptr<CompositeShape> shape=drawRVO(sim);
  drawer.addShape(shape);
  drawer.addCamera2D(ext);
  drawer.clearLight();
  bool step=false;
  drawer.setKeyFunc([&](GLFWwindow*,int key,int,int action,int,bool captured) {
    if(captured)
      return;
    else if(key==GLFW_KEY_R && action==GLFW_PRESS)
      step=!step;
  });
  drawer.setFrameFunc([&](std::shared_ptr<SceneNode>&) {
    if(step) {
      frm();
      drawRVO(sim,shape);
    }
  });
  drawer.mainLoop();
}
void drawRVOApp(float ext,RVOSimulator& sim) {
  using namespace DRAWER;
  drawRVOApp(0,NULL,ext,sim,[&]() {
    sim.updateAgentTargets();
    sim.optimize(false,false);
  });
}
}

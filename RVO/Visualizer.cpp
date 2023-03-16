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
using namespace DRAWER;
float COLOR_AGT[3]= {.6,.6,.6};
float COLOR_OBS[3]= {.0,.0,.0};
float COLOR_VEL[3]= {.5,.0,.0};
std::shared_ptr<CompositeShape> drawRVOPosition(const RVOSimulator& sim,std::shared_ptr<CompositeShape> shapesInput) {
  std::shared_ptr<CompositeShape> shapes=shapesInput?shapesInput:std::shared_ptr<CompositeShape>(new CompositeShape);
  if(!shapesInput) {
    for(int i=0; i<sim.getNrAgent(); i++) {
      std::shared_ptr<Bullet3DShape> agent(new Bullet3DShape);
      std::shared_ptr<MeshShape> circle=makeCircle(16,true,Eigen::Matrix<GLfloat,2,1>::Zero(),(GLfloat)sim.getAgentRadius(i));
      circle->setColorAmbient(GL_TRIANGLES,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2]);
      circle->setColor(GL_TRIANGLES,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2]);
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
      obs->setColor(GL_TRIANGLES,COLOR_OBS[0],COLOR_OBS[1],COLOR_OBS[2]);
      shapes->addShape(obs);
    }
  }
  for(int i=0; i<sim.getNrAgent(); i++)
    std::dynamic_pointer_cast<Bullet3DShape>(shapes->getChild(i))->
    setLocalTranslate(Eigen::Matrix<GLfloat,3,1>((GLfloat)sim.getAgentPosition(i)[0],(GLfloat)sim.getAgentPosition(i)[1],0));
  return shapes;
}
std::shared_ptr<MeshShape> drawRVOVelocity(const RVOSimulator& sim,std::shared_ptr<MeshShape> shapesInput) {
  std::shared_ptr<MeshShape> shapes=shapesInput?shapesInput:std::shared_ptr<MeshShape>(new MeshShape);
  RVOSimulator::Mat2XT pss=sim.getAgentPositions();
  RVOSimulator::Mat2XT vss=sim.getAgentVelocities()+pss;
  if(!shapesInput) {
    for(int i=0; i<sim.getNrAgent(); i++) {
      shapes->addVertex(Eigen::Matrix<float,3,1>(pss(0,i),pss(1,i),0));
      shapes->addVertex(Eigen::Matrix<float,3,1>(vss(0,i),vss(1,i),0));
      shapes->addIndexSingle(i*2+0);
      shapes->addIndexSingle(i*2+1);
    }
    shapes->setMode(GL_LINES);
    shapes->setColor(GL_LINES,COLOR_VEL[0],COLOR_VEL[1],COLOR_VEL[2]);
    shapes->setLineWidth(5);
  }
  for(int i=0; i<sim.getNrAgent(); i++) {
    shapes->setVertex(i*2+0,Eigen::Matrix<float,3,1>(pss(0,i),pss(1,i),0));
    shapes->setVertex(i*2+1,Eigen::Matrix<float,3,1>(vss(0,i),vss(1,i),0));
  }
  return shapes;
}
std::shared_ptr<MeshShape> drawLines(const std::vector<Eigen::Matrix<LSCALAR,2,1>>& vss,const Eigen::Matrix<GLfloat,3,1>& color) {
  std::shared_ptr<MeshShape> mesh(new MeshShape);
  for(int i=0; i<(int)vss.size(); i++) {
    mesh->addVertex(Eigen::Matrix<GLfloat,3,1>((GLfloat)vss[i][0],(GLfloat)vss[i][1],0));
    mesh->addIndexSingle(i);
  }
  mesh->setMode(GL_LINES);
  mesh->setColor(GL_LINES,color[0],color[1],color[2]);
  mesh->setLineWidth(5);
  return mesh;
}
void drawVisibleApp(int argc,char** argv,float ext,const RVOSimulator& sim,
                    const std::vector<Eigen::Matrix<LSCALAR,2,1>>& vss,
                    const std::vector<Eigen::Matrix<LSCALAR,2,1>>& nvss) {
  Drawer drawer(argc,argv);
  drawer.addPlugin(std::shared_ptr<Plugin>(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat")));
  drawer.addPlugin(std::shared_ptr<Plugin>(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",drawer.FPS())));
  drawer.addShape(drawRVOPosition(sim));
  if(!vss.empty())
    drawer.addShape(drawLines(vss,Eigen::Matrix<GLfloat,3,1>(.7,.2,.2)));
  if(!nvss.empty())
    drawer.addShape(drawLines(nvss,Eigen::Matrix<GLfloat,3,1>(.2,.7,.7)));
  drawer.addCamera2D(ext);
  drawer.clearLight();
  drawer.mainLoop();
}
void drawRVOApp(int argc,char** argv,GLfloat ext,const RVOSimulator& sim,std::function<void()> frm) {
  Drawer drawer(argc,argv);
  drawer.addPlugin(std::shared_ptr<Plugin>(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat")));
  drawer.addPlugin(std::shared_ptr<Plugin>(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",drawer.FPS())));
  std::shared_ptr<CompositeShape> agent=drawRVOPosition(sim);
  std::shared_ptr<MeshShape> vel=drawRVOVelocity(sim);
  drawer.addShape(agent);
  drawer.addShape(vel);
  drawer.addCamera2D(ext);
  drawer.clearLight();
  bool step=false;
  drawer.setKeyFunc([&](GLFWwindow*,int key,int,int action,int,bool captured) {
    if(captured)
      return;
    if(key==GLFW_KEY_R && action==GLFW_PRESS)
      step=!step;
  });
  drawer.setFrameFunc([&](std::shared_ptr<SceneNode>&) {
    if(step)
      frm();
    drawRVOPosition(sim,agent);
    drawRVOVelocity(sim,vel);
  });
  drawer.mainLoop();
}
void drawRVOApp(int argc,char** argv,GLfloat ext,const MultiRVOSimulator& sim,std::function<void()> frm) {
  Drawer drawer(argc,argv);
  drawer.addPlugin(std::shared_ptr<Plugin>(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat")));
  drawer.addPlugin(std::shared_ptr<Plugin>(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",drawer.FPS())));
  std::shared_ptr<CompositeShape> agent=drawRVOPosition(sim.getSubSimulator(0));
  std::shared_ptr<MeshShape> vel=drawRVOVelocity(sim.getSubSimulator(0));
  drawer.addShape(agent);
  drawer.addShape(vel);
  drawer.addCamera2D(ext);
  drawer.clearLight();
  bool step=false;
  int id=0;
  drawer.setKeyFunc([&](GLFWwindow*,int key,int,int action,int,bool captured) {
    if(captured)
      return;
    if(key==GLFW_KEY_R && action==GLFW_PRESS)
      step=!step;
    if(key==GLFW_KEY_D && action==GLFW_PRESS) {
      id=(id+1)%sim.getBatchSize();
      drawRVOPosition(sim.getSubSimulator(id),agent);
      drawRVOVelocity(sim.getSubSimulator(id),vel);
    }
    if(key==GLFW_KEY_A && action==GLFW_PRESS) {
      id=(id+sim.getBatchSize()-1)%sim.getBatchSize();
      drawRVOPosition(sim.getSubSimulator(id),agent);
      drawRVOVelocity(sim.getSubSimulator(id),vel);
    }
  });
  drawer.setFrameFunc([&](std::shared_ptr<SceneNode>&) {
    if(step)
      frm();
    drawRVOPosition(sim.getSubSimulator(id),agent);
    drawRVOVelocity(sim.getSubSimulator(id),vel);
  });
  drawer.mainLoop();
}
void drawRVOApp(float ext,RVOSimulator& sim) {
  drawRVOApp(0,NULL,ext,sim,[&]() {
    sim.updateAgentTargets();
    sim.optimize(false,false);
  });
}
void drawRVOApp(float ext,MultiRVOSimulator& sim) {
  drawRVOApp(0,NULL,ext,sim,[&]() {
    sim.updateAgentTargets();
    sim.optimize(false,false);
  });
}
}

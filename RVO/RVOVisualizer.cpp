#include "RVOVisualizer.h"
#include <TinyVisualizer/Drawer.h>
#include <TinyVisualizer/Camera2D.h>
#include <TinyVisualizer/MakeMesh.h>
#include <TinyVisualizer/MeshShape.h>
#include <TinyVisualizer/Bullet3DShape.h>
#include <TinyVisualizer/CompositeShape.h>
#include <TinyVisualizer/CaptureGIFPlugin.h>
#include <TinyVisualizer/CameraExportPlugin.h>

namespace RVO {
float COLOR_AGT[3]= {200/255.,143/255., 29/255.};
float COLOR_OBS[3]= {000/255.,000/255.,000/255.};
float COLOR_VEL[3]= {120/255.,000/255.,000/255.};
std::vector<std::tuple<Eigen::Matrix<float,2,1>,Eigen::Matrix<float,2,1>,Eigen::Matrix<float,3,1>>> qss;
std::vector<std::tuple<Eigen::Matrix<float,2,1>,Eigen::Matrix<float,2,1>,Eigen::Matrix<float,3,1>>> lss;
void RVOVisualizer::drawQuad(Eigen::Matrix<float,2,1> from,Eigen::Matrix<float,2,1> to,Eigen::Matrix<float,3,1> color) {
  qss.push_back(std::make_tuple(from,to,color));
}
void RVOVisualizer::drawLine(Eigen::Matrix<float,2,1> from,Eigen::Matrix<float,2,1> to,Eigen::Matrix<float,3,1> color) {
  lss.push_back(std::make_tuple(from,to,color));
}
void RVOVisualizer::clearQuad() {
  qss.clear();
}
void RVOVisualizer::clearLine() {
  lss.clear();
}
std::shared_ptr<CompositeShape> RVOVisualizer::drawRVOPosition(const RVOSimulator& sim,std::shared_ptr<CompositeShape> shapesInput) {
  std::shared_ptr<CompositeShape> shapes=shapesInput?shapesInput:std::shared_ptr<CompositeShape>(new CompositeShape);
  if(!shapesInput) {
    for(int i=0; i<sim.getNrAgent(); i++) {
      std::shared_ptr<Bullet3DShape> agent(new Bullet3DShape);
      std::shared_ptr<MeshShape> circle=makeCircle(16,true,Eigen::Matrix<float,2,1>::Zero(),(float)sim.getAgentRadius(i));
      circle->setColor(GL_TRIANGLE_FAN,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2]);
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
        obs->addVertex(Eigen::Matrix<float,3,1>((float)pos[0  ][0],(float)pos[0  ][1],0));
        obs->addVertex(Eigen::Matrix<float,3,1>((float)pos[j+1][0],(float)pos[j+1][1],0));
        obs->addVertex(Eigen::Matrix<float,3,1>((float)pos[j+2][0],(float)pos[j+2][1],0));
      }
      obs->setMode(GL_TRIANGLES);
      obs->setColor(GL_TRIANGLES,COLOR_OBS[0],COLOR_OBS[1],COLOR_OBS[2]);
      shapes->addShape(obs);
    }
  }
  for(int i=0; i<sim.getNrAgent(); i++)
    std::dynamic_pointer_cast<Bullet3DShape>(shapes->getChild(i))->
    setLocalTranslate(Eigen::Matrix<float,3,1>((float)sim.getAgentPosition(i)[0],(float)sim.getAgentPosition(i)[1],0));
  return shapes;
}
std::shared_ptr<MeshShape> RVOVisualizer::drawRVOVelocity(const RVOSimulator& sim,std::shared_ptr<MeshShape> shapesInput) {
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
std::shared_ptr<MeshShape> RVOVisualizer::drawLines(const std::vector<Eigen::Matrix<LSCALAR,2,1>>& vss,const Eigen::Matrix<float,3,1>& color) {
  std::shared_ptr<MeshShape> mesh(new MeshShape);
  for(int i=0; i<(int)vss.size(); i++) {
    mesh->addVertex(Eigen::Matrix<float,3,1>((float)vss[i][0],(float)vss[i][1],0));
    mesh->addIndexSingle(i);
  }
  mesh->setMode(GL_LINES);
  mesh->setColor(GL_LINES,color[0],color[1],color[2]);
  mesh->setLineWidth(5);
  return mesh;
}
std::shared_ptr<CompositeShape> RVOVisualizer::drawLines() {
  std::shared_ptr<CompositeShape> lines(new CompositeShape);
  for(int i=0; i<(int)lss.size(); i++) {
    std::shared_ptr<MeshShape> line(new MeshShape);
    line->addVertex(Eigen::Matrix<float,3,1>((float)std::get<0>(lss[i])[0],(float)std::get<0>(lss[i])[1],0));
    line->addVertex(Eigen::Matrix<float,3,1>((float)std::get<1>(lss[i])[0],(float)std::get<1>(lss[i])[1],0));
    line->addIndexSingle(0);
    line->addIndexSingle(1);
    line->setMode(GL_LINES);
    line->setColor(GL_LINES,std::get<2>(lss[i])[0],std::get<2>(lss[i])[1],std::get<2>(lss[i])[2]);
    line->setLineWidth(5);
    lines->addShape(line);
  }
  return lines;
}
std::shared_ptr<CompositeShape> RVOVisualizer::drawQuads() {
  std::shared_ptr<CompositeShape> quads(new CompositeShape);
  for(int i=0; i<(int)qss.size(); i++) {
    std::shared_ptr<MeshShape> quad(new MeshShape);
    quad->addVertex(Eigen::Matrix<float,3,1>((float)std::get<0>(qss[i])[0],(float)std::get<0>(qss[i])[1],0));
    quad->addVertex(Eigen::Matrix<float,3,1>((float)std::get<1>(qss[i])[0],(float)std::get<0>(qss[i])[1],0));
    quad->addVertex(Eigen::Matrix<float,3,1>((float)std::get<1>(qss[i])[0],(float)std::get<1>(qss[i])[1],0));
    quad->addVertex(Eigen::Matrix<float,3,1>((float)std::get<0>(qss[i])[0],(float)std::get<1>(qss[i])[1],0));
    quad->addIndexSingle(0);
    quad->addIndexSingle(1);
    quad->addIndexSingle(2);
    quad->addIndexSingle(3);
    quad->setMode(GL_TRIANGLE_FAN);
    quad->setColor(GL_TRIANGLE_FAN,std::get<2>(qss[i])[0],std::get<2>(qss[i])[1],std::get<2>(qss[i])[2]);
    quads->addShape(quad);
  }
  return quads;
}
void RVOVisualizer::drawVisibleApp(int argc,char** argv,float ext,const RVOSimulator& sim,
                                   const std::vector<Eigen::Matrix<LSCALAR,2,1>>& vss,
                                   const std::vector<Eigen::Matrix<LSCALAR,2,1>>& nvss) {
  Drawer drawer(argc,argv);
  drawer.addPlugin(std::shared_ptr<Plugin>(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat")));
  drawer.addPlugin(std::shared_ptr<Plugin>(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",drawer.FPS())));
  drawer.addShape(drawRVOPosition(sim));
  if(!vss.empty())
    drawer.addShape(drawLines(vss,Eigen::Matrix<float,3,1>(.7,.2,.2)));
  if(!nvss.empty())
    drawer.addShape(drawLines(nvss,Eigen::Matrix<float,3,1>(.2,.7,.7)));
  drawer.addCamera2D(ext);
  drawer.clearLight();
  drawer.mainLoop();
}
void RVOVisualizer::drawRVO(int argc,char** argv,float ext,const RVOSimulator& sim,std::function<void()> frm,PythonCallback* cb) {
  Drawer drawer(argc,argv);
  if(cb)
    drawer.setPythonCallback(cb);
  drawer.addPlugin(std::shared_ptr<Plugin>(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat")));
  drawer.addPlugin(std::shared_ptr<Plugin>(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",drawer.FPS())));
  std::shared_ptr<CompositeShape> agent=drawRVOPosition(sim);
  std::shared_ptr<MeshShape> vel=drawRVOVelocity(sim);
  agent->addShape(drawLines());
  agent->addShape(drawQuads());
  drawer.addShape(agent);
  drawer.addCamera2D(ext);
  drawer.clearLight();
  bool step=false;
  drawer.setKeyFunc([&](GLFWwindow*,int key,int,int action,int,bool captured) {
    if(captured)
      return;
    if(key==GLFW_KEY_R && action==GLFW_PRESS)
      step=!step;
    if(key==GLFW_KEY_W && action==GLFW_PRESS) {
      if(agent->contain(vel))
        agent->removeChild(vel);
      else agent->addShape(vel);
    }
  });
  drawer.setFrameFunc([&](std::shared_ptr<SceneNode>&) {
    if(step)
      frm();
    drawRVOPosition(sim,agent);
    drawRVOVelocity(sim,vel);
  });
  drawer.mainLoop();
}
void RVOVisualizer::drawRVO(int argc,char** argv,float ext,const MultiRVOSimulator& sim,std::function<void()> frm,PythonCallback* cb) {
  Drawer drawer(argc,argv);
  if(cb)
    drawer.setPythonCallback(cb);
  drawer.addPlugin(std::shared_ptr<Plugin>(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat")));
  drawer.addPlugin(std::shared_ptr<Plugin>(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",drawer.FPS())));
  std::shared_ptr<CompositeShape> agent=drawRVOPosition(sim.getSubSimulator(0));
  std::shared_ptr<MeshShape> vel=drawRVOVelocity(sim.getSubSimulator(0));
  agent->addShape(drawLines());
  agent->addShape(drawQuads());
  drawer.addShape(agent);
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
    if(key==GLFW_KEY_W && action==GLFW_PRESS) {
      if(agent->contain(vel))
        agent->removeChild(vel);
      else agent->addShape(vel);
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
void RVOVisualizer::drawRVO(float ext,RVOSimulator& sim) {
  RVOVisualizer::drawRVO(0,NULL,ext,sim,[&]() {
    sim.updateAgentTargets();
    sim.optimize(false,false);
  },NULL);
}
void RVOVisualizer::drawRVO(float ext,MultiRVOSimulator& sim) {
  RVOVisualizer::drawRVO(0,NULL,ext,sim,[&]() {
    sim.updateAgentTargets();
    sim.optimize(false,false);
  },NULL);
}
void RVOVisualizer::drawRVO(float ext,RVOSimulator& sim,PythonCallback* cb) {
  RVOVisualizer::drawRVO(0,NULL,ext,sim,[&]() {
    sim.updateAgentTargets();
    sim.optimize(false,false);
  },cb);
}
void RVOVisualizer::drawRVO(float ext,MultiRVOSimulator& sim,PythonCallback* cb) {
  RVOVisualizer::drawRVO(0,NULL,ext,sim,[&]() {
    sim.updateAgentTargets();
    sim.optimize(false,false);
  },cb);
}
}

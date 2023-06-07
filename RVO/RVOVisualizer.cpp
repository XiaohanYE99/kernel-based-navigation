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
float COLOR_VIS[3]= {000/255.,255/255.,000/255.};
bool quadsUpdate=true;
bool linesUpdate=true;
std::vector<std::tuple<Eigen::Matrix<float,2,1>,Eigen::Matrix<float,2,1>,Eigen::Matrix<float,3,1>>> qss;
std::vector<std::tuple<Eigen::Matrix<float,2,1>,Eigen::Matrix<float,2,1>,Eigen::Matrix<float,3,1>>> lss;
void RVOVisualizer::drawQuad(Eigen::Matrix<float,2,1> from,Eigen::Matrix<float,2,1> to,Eigen::Matrix<float,3,1> color) {
  qss.push_back(std::make_tuple(from,to,color));
  quadsUpdate=true;
}
void RVOVisualizer::drawLine(Eigen::Matrix<float,2,1> from,Eigen::Matrix<float,2,1> to,Eigen::Matrix<float,3,1> color) {
  lss.push_back(std::make_tuple(from,to,color));
  linesUpdate=true;
}
void RVOVisualizer::drawVisibility(const VisibilityGraph& graph,const Eigen::Matrix<LSCALAR,2,1> p) {
  for(const auto& line:graph.lines(p))
    drawLine(line.first.template cast<float>(),
             line.second.template cast<float>(),
             Eigen::Matrix<float,3,1>(COLOR_VIS[0],COLOR_VIS[1],COLOR_VIS[2]));
}
void RVOVisualizer::drawVisibility(const VisibilityGraph& graph,int id) {
  for(const auto& line:graph.lines(id))
    drawLine(line.first.template cast<float>(),
             line.second.template cast<float>(),
             Eigen::Matrix<float,3,1>(COLOR_VIS[0],COLOR_VIS[1],COLOR_VIS[2]));
}
void RVOVisualizer::clearQuad() {
  qss.clear();
  quadsUpdate=true;
}
void RVOVisualizer::clearLine() {
  lss.clear();
  linesUpdate=true;
}
int RVOVisualizer::getNrQuads() {
  return (int)qss.size();
}
void RVOVisualizer::setNrQuads(int nr) {
  qss.resize(nr);
  quadsUpdate=true;
}
int RVOVisualizer::getNrLines() {
  return (int)lss.size();
}
void RVOVisualizer::setNrLines(int nr) {
  lss.resize(nr);
  linesUpdate=true;
}
void RVOVisualizer::drawObstacle(const RVOSimulator& sim,std::shared_ptr<CompositeShape> shapes) {
  std::shared_ptr<CompositeShape> obss(new CompositeShape);
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
    obss->addShape(obs);
  }
  shapes->addShape(obss);
}
std::shared_ptr<CompositeShape> RVOVisualizer::drawRVOPosition(const RVOSimulator& sim,std::shared_ptr<CompositeShape> shapesInput) {
  std::shared_ptr<CompositeShape> shapes=shapesInput?shapesInput:std::shared_ptr<CompositeShape>(new CompositeShape);
  if(!shapesInput)
    drawObstacle(sim,shapes);
  //need more children
  while(shapes->numChildren()<sim.getNrAgent()+1) {
    std::shared_ptr<Bullet3DShape> agent(new Bullet3DShape);
    std::shared_ptr<MeshShape> circle=makeCircle(16,true,Eigen::Matrix<float,2,1>::Zero(),1);
    circle->setColor(GL_TRIANGLE_FAN,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2]);
    agent->addShape(circle);
    shapes->addShape(agent);
  }
  //less children
  while(shapes->numChildren()>sim.getNrAgent()+1)
    shapes->removeChild(shapes->getChild(shapes->numChildren()-1));
  //update translation
  Eigen::Matrix<float,4,4> t;
  for(int i=0; i<sim.getNrAgent(); i++) {
    t=Eigen::Matrix<float,4,4>::Identity();
    t(0,0)*=sim.getAgentRadius(i);
    t(1,1)*=sim.getAgentRadius(i);
    t(0,3)=(float)sim.getAgentPosition(i)[0];
    t(1,3)=(float)sim.getAgentPosition(i)[1];
    std::dynamic_pointer_cast<Bullet3DShape>(shapes->getChild(i+1))->setLocalTransform(t);
  }
  return shapes;
}
std::shared_ptr<CompositeShape> RVOVisualizer::drawRVOPosition(int frameId,const std::vector<Trajectory>& trajectories,const RVOSimulator& sim,std::shared_ptr<CompositeShape> shapesInput) {
  std::shared_ptr<CompositeShape> shapes=shapesInput?shapesInput:std::shared_ptr<CompositeShape>(new CompositeShape);
  if(!shapesInput)
    drawObstacle(sim,shapes);
  std::pair<Trajectory::Mat2XT,Trajectory::Vec> frame=SourceSink::getAgentPositions(frameId,trajectories);
  int nrAgent=frame.first.cols();
  //need more children
  while(shapes->numChildren()<nrAgent+1) {
    std::shared_ptr<Bullet3DShape> agent(new Bullet3DShape);
    std::shared_ptr<MeshShape> circle=makeCircle(16,true,Eigen::Matrix<float,2,1>::Zero(),1);
    circle->setColor(GL_TRIANGLE_FAN,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2]);
    agent->addShape(circle);
    shapes->addShape(agent);
  }
  //less children
  while(shapes->numChildren()>nrAgent+1)
    shapes->removeChild(shapes->getChild(shapes->numChildren()-1));
  //update translation
  Eigen::Matrix<float,4,4> t;
  for(int i=0; i<nrAgent; i++) {
    t=Eigen::Matrix<float,4,4>::Identity();
    t(0,0)*=frame.second[i];
    t(1,1)*=frame.second[i];
    t(0,3)=(float)frame.first.col(i)[0];
    t(1,3)=(float)frame.first.col(i)[1];
    std::dynamic_pointer_cast<Bullet3DShape>(shapes->getChild(i+1))->setLocalTransform(t);
  }
  return shapes;
}
std::shared_ptr<MeshShape> RVOVisualizer::drawRVOVelocity(const RVOSimulator& sim,std::shared_ptr<MeshShape> shapesInput) {
  std::shared_ptr<MeshShape> shapes=shapesInput?shapesInput:std::shared_ptr<MeshShape>(new MeshShape);
  RVOSimulator::Mat2XT pss=sim.getAgentPositions();
  RVOSimulator::Mat2XT vss=sim.getAgentVelocities()+pss;
  if(!shapesInput || shapesInput->nrVertex()!=sim.getNrAgent()*2) {
    shapes->clear();
    for(int i=0; i<sim.getNrAgent(); i++) {
      shapes->addVertex(Eigen::Matrix<float,3,1>(pss(0,i),pss(1,i),0));
      shapes->addVertex(Eigen::Matrix<float,3,1>(vss(0,i),vss(1,i),0));
      shapes->addIndexSingle(i*2+0);
      shapes->addIndexSingle(i*2+1);
    }
    shapes->setMode(GL_LINES);
    shapes->setColor(GL_LINES,COLOR_VEL[0],COLOR_VEL[1],COLOR_VEL[2]);
    shapes->setLineWidth(5);
  } else {
    for(int i=0; i<sim.getNrAgent(); i++) {
      shapes->setVertex(i*2+0,Eigen::Matrix<float,3,1>(pss(0,i),pss(1,i),0));
      shapes->setVertex(i*2+1,Eigen::Matrix<float,3,1>(vss(0,i),vss(1,i),0));
    }
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
std::shared_ptr<CompositeShape> RVOVisualizer::drawLines(std::shared_ptr<CompositeShape> linesRef) {
  std::shared_ptr<CompositeShape> lines;
  if(linesRef)
    lines=linesRef;
  else lines.reset(new CompositeShape);
  if(!linesUpdate)
    return lines;
  while(lines->numChildren()>0)
    lines->removeChild(lines->getChild(0));
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
  linesUpdate=false;
  return lines;
}
std::shared_ptr<CompositeShape> RVOVisualizer::drawQuads(std::shared_ptr<CompositeShape> quadsRef) {
  std::shared_ptr<CompositeShape> quads;
  if(quadsRef)
    quads=quadsRef;
  else quads.reset(new CompositeShape);
  if(!quadsUpdate)
    return quads;
  while(quads->numChildren()>0)
    quads->removeChild(quads->getChild(0));
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
  quadsUpdate=false;
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
  std::shared_ptr<CompositeShape> agent=drawRVOPosition(sim),lines,quads;
  std::shared_ptr<MeshShape> vel=drawRVOVelocity(sim);
  agent->addShape(lines=drawLines(lines));
  agent->addShape(quads=drawQuads(quads));
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
    drawLines(lines);
    drawQuads(quads);
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
  std::shared_ptr<CompositeShape> agent=drawRVOPosition(sim.getSubSimulator(0)),lines,quads;
  std::shared_ptr<MeshShape> vel=drawRVOVelocity(sim.getSubSimulator(0));
  agent->addShape(lines=drawLines(lines));
  agent->addShape(quads=drawQuads(quads));
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
    drawLines(lines);
    drawQuads(quads);
    drawRVOPosition(sim.getSubSimulator(id),agent);
    drawRVOVelocity(sim.getSubSimulator(id),vel);
  });
  drawer.mainLoop();
}
void RVOVisualizer::drawRVO(int argc,char** argv,float ext,const std::vector<Trajectory>& trajs,const RVOSimulator& sim,std::function<void()> frm,PythonCallback* cb) {
  Drawer drawer(argc,argv);
  if(cb)
    drawer.setPythonCallback(cb);
  int frameId=0;
  drawer.addPlugin(std::shared_ptr<Plugin>(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat")));
  drawer.addPlugin(std::shared_ptr<Plugin>(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",drawer.FPS())));
  std::shared_ptr<CompositeShape> agent=drawRVOPosition(frameId,trajs,sim),lines,quads;
  agent->addShape(lines=drawLines(lines));
  agent->addShape(quads=drawQuads(quads));
  drawer.addShape(agent);
  drawer.addCamera2D(ext);
  drawer.clearLight();
  bool step=false;
  drawer.setKeyFunc([&](GLFWwindow*,int key,int,int action,int,bool captured) {
    if(captured)
      return;
    if(key==GLFW_KEY_R && action==GLFW_PRESS)
      step=!step;
    if(key==GLFW_KEY_W && action==GLFW_PRESS)
      frameId=0;
  });
  drawer.setFrameFunc([&](std::shared_ptr<SceneNode>&) {
    if(step) {
      frameId++;
      frm();
    }
    drawLines(lines);
    drawQuads(quads);
    drawRVOPosition(frameId,trajs,sim,agent);
  });
  drawer.mainLoop();
}
void RVOVisualizer::drawRVO(int argc,char** argv,float ext,const std::vector<std::vector<Trajectory>>& trajs,const MultiRVOSimulator& sim,std::function<void()> frm,PythonCallback* cb) {
  Drawer drawer(argc,argv);
  if(cb)
    drawer.setPythonCallback(cb);
  int frameId=0;
  drawer.addPlugin(std::shared_ptr<Plugin>(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat")));
  drawer.addPlugin(std::shared_ptr<Plugin>(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",drawer.FPS())));
  std::shared_ptr<CompositeShape> agent=drawRVOPosition(frameId,trajs[0],sim.getSubSimulator(0)),lines,quads;
  agent->addShape(lines=drawLines(lines));
  agent->addShape(quads=drawQuads(quads));
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
      id=(id+1)%(int)trajs.size();
      drawRVOPosition(frameId,trajs[id],sim.getSubSimulator(0),agent);
    }
    if(key==GLFW_KEY_A && action==GLFW_PRESS) {
      id=(id+(int)trajs.size()-1)%(int)trajs.size();
      drawRVOPosition(frameId,trajs[id],sim.getSubSimulator(0),agent);
    }
    if(key==GLFW_KEY_W && action==GLFW_PRESS)
      frameId=0;
  });
  drawer.setFrameFunc([&](std::shared_ptr<SceneNode>&) {
    if(step) {
      frameId++;
      frm();
    }
    drawLines(lines);
    drawQuads(quads);
    drawRVOPosition(frameId,trajs[id],sim.getSubSimulator(0),agent);
  });
  drawer.mainLoop();
}
//convenient functions
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
  RVOVisualizer::drawRVO(0,NULL,ext,sim,[&]() {},cb);
}
void RVOVisualizer::drawRVO(float ext,MultiRVOSimulator& sim,PythonCallback* cb) {
  RVOVisualizer::drawRVO(0,NULL,ext,sim,[&]() {},cb);
}
void RVOVisualizer::drawRVO(float ext,const std::vector<Trajectory>& trajs,const RVOSimulator& sim) {
  RVOVisualizer::drawRVO(0,NULL,ext,trajs,sim,[&]() {});
}
void RVOVisualizer::drawRVO(float ext,const std::vector<std::vector<Trajectory>>& trajs,const MultiRVOSimulator& sim) {
  RVOVisualizer::drawRVO(0,NULL,ext,trajs,sim,[&]() {});
}
void RVOVisualizer::drawRVO(float ext,const std::vector<Trajectory>& trajs,const RVOSimulator& sim,PythonCallback* cb) {
  RVOVisualizer::drawRVO(0,NULL,ext,trajs,sim,[&]() {},cb);
}
void RVOVisualizer::drawRVO(float ext,const std::vector<std::vector<Trajectory>>& trajs,const MultiRVOSimulator& sim,PythonCallback* cb) {
  RVOVisualizer::drawRVO(0,NULL,ext,trajs,sim,[&]() {},cb);
}
}

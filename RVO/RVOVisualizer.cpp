#include "RVOVisualizer.h"
#include <TinyVisualizer/Drawer.h>
#include <TinyVisualizer/Camera2D.h>
#include <TinyVisualizer/MakeMesh.h>
#include <TinyVisualizer/MeshShape.h>
#include <TinyVisualizer/Bullet3DShape.h>
#include <TinyVisualizer/CompositeShape.h>
#include <TinyVisualizer/ImGuiPlugin.h>
#include <imgui/imgui.h>

namespace RVO {
float COLOR_AGT[3]= {200/255.,143/255., 29/255.};
float COLOR_OBS[3]= {000/255.,000/255.,000/255.};
float COLOR_VEL[3]= {120/255.,000/255.,000/255.};
float COLOR_VIS[3]= {000/255.,255/255.,000/255.};
//RVOPythonCallback
void RVOPythonCallback::mouse(int button,int action,int mods) {
  if(_mouse)
    _mouse(button,action,mods);
}
void RVOPythonCallback::wheel(double xoffset,double yoffset) {
  if(_wheel)
    _wheel(xoffset,yoffset);
}
void RVOPythonCallback::motion(double x,double y) {
  if(_motion)
    _motion(x,y);
}
void RVOPythonCallback::key(int key,int scan,int action,int mods) {
  if(_key)
    _key(key,scan,action,mods);
}
void RVOPythonCallback::frame(std::shared_ptr<SceneNode>& root) {
  if(_frame)
    _frame();
}
void RVOPythonCallback::draw() {
  if(_draw)
    _draw();
}
void RVOPythonCallback::setup() {
  if(_setup)
    _setup();
}
//RVOVisualizer
void RVOVisualizer::clearSourceColor() {
  _css.clear();
}
void RVOVisualizer::setSourceColor(unsigned short sid,Eigen::Matrix<float,3,1> color) {
  _css[sid]=color;
}
void RVOVisualizer::drawQuad(Eigen::Matrix<float,2,1> from,Eigen::Matrix<float,2,1> to,Eigen::Matrix<float,3,1> color) {
  _qss.push_back(std::make_tuple(from,to,color));
  _quadsUpdate=true;
}
void RVOVisualizer::drawLine(Eigen::Matrix<float,2,1> from,Eigen::Matrix<float,2,1> to,Eigen::Matrix<float,3,1> color) {
  _lss.push_back(std::make_tuple(from,to,color));
  _linesUpdate=true;
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
  _qss.clear();
  _quadsUpdate=true;
}
void RVOVisualizer::clearLine() {
  _lss.clear();
  _linesUpdate=true;
}
int RVOVisualizer::getNrQuads() {
  return (int)_qss.size();
}
void RVOVisualizer::setNrQuads(int nr) {
  _qss.resize(nr);
  _quadsUpdate=true;
}
int RVOVisualizer::getNrLines() {
  return (int)_lss.size();
}
void RVOVisualizer::setNrLines(int nr) {
  _lss.resize(nr);
  _linesUpdate=true;
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
    obs->setColorDiffuse(GL_TRIANGLES,COLOR_OBS[0],COLOR_OBS[1],COLOR_OBS[2]);
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
    circle->setColorDiffuse(GL_TRIANGLE_FAN,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2]);
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
    //set agent color
    std::shared_ptr<Bullet3DShape> shape=std::dynamic_pointer_cast<Bullet3DShape>(shapes->getChild(i+1));
    unsigned short sid=SourceSink::extractSourceId(sim.getAgentId(i));
    const auto it=_css.find(sid);
    if(it==_css.end())
      shape->setColorDiffuse(GL_TRIANGLE_FAN,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2]);
    else shape->setColorDiffuse(GL_TRIANGLE_FAN,it->second.x(),it->second.y(),it->second.z());
    shape->setLocalTransform(t);
  }
  return shapes;
}
std::shared_ptr<CompositeShape> RVOVisualizer::drawRVOPosition(int frameId,const std::vector<Trajectory>& trajectories,const RVOSimulator& sim,std::shared_ptr<CompositeShape> shapesInput) {
  std::shared_ptr<CompositeShape> shapes=shapesInput?shapesInput:std::shared_ptr<CompositeShape>(new CompositeShape);
  if(!shapesInput)
    drawObstacle(sim,shapes);
  SourceSink::Frame frame=SourceSink::getAgentPositions(frameId,trajectories);
  int nrAgent=std::get<0>(frame).cols();
  //need more children
  while(shapes->numChildren()<nrAgent+1) {
    std::shared_ptr<Bullet3DShape> agent(new Bullet3DShape);
    std::shared_ptr<MeshShape> circle=makeCircle(16,true,Eigen::Matrix<float,2,1>::Zero(),1);
    circle->setColorDiffuse(GL_TRIANGLE_FAN,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2]);
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
    t(0,0)*=std::get<1>(frame)[i];
    t(1,1)*=std::get<1>(frame)[i];
    t(0,3)=(float)std::get<0>(frame).col(i)[0];
    t(1,3)=(float)std::get<0>(frame).col(i)[1];
    //set agent color
    std::shared_ptr<Bullet3DShape> shape=std::dynamic_pointer_cast<Bullet3DShape>(shapes->getChild(i+1));
    unsigned short sid=std::get<2>(frame)[i];
    const auto it=_css.find(sid);
    if(it==_css.end())
      shape->setColorDiffuse(GL_TRIANGLE_FAN,COLOR_AGT[0],COLOR_AGT[1],COLOR_AGT[2]);
    else shape->setColorDiffuse(GL_TRIANGLE_FAN,it->second.x(),it->second.y(),it->second.z());
    shape->setLocalTransform(t);
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
    shapes->setColorDiffuse(GL_LINES,COLOR_VEL[0],COLOR_VEL[1],COLOR_VEL[2]);
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
  mesh->setColorDiffuse(GL_LINES,color[0],color[1],color[2]);
  mesh->setLineWidth(5);
  return mesh;
}
std::shared_ptr<CompositeShape> RVOVisualizer::drawLines(std::shared_ptr<CompositeShape> linesRef) {
  std::shared_ptr<CompositeShape> lines;
  if(linesRef)
    lines=linesRef;
  else lines.reset(new CompositeShape);
  if(!_linesUpdate)
    return lines;
  while(lines->numChildren()>0)
    lines->removeChild(lines->getChild(0));
  for(int i=0; i<(int)_lss.size(); i++) {
    std::shared_ptr<MeshShape> line(new MeshShape);
    line->addVertex(Eigen::Matrix<float,3,1>((float)std::get<0>(_lss[i])[0],(float)std::get<0>(_lss[i])[1],0));
    line->addVertex(Eigen::Matrix<float,3,1>((float)std::get<1>(_lss[i])[0],(float)std::get<1>(_lss[i])[1],0));
    line->addIndexSingle(0);
    line->addIndexSingle(1);
    line->setMode(GL_LINES);
    line->setColorDiffuse(GL_LINES,std::get<2>(_lss[i])[0],std::get<2>(_lss[i])[1],std::get<2>(_lss[i])[2]);
    line->setLineWidth(5);
    lines->addShape(line);
  }
  _linesUpdate=false;
  return lines;
}
std::shared_ptr<CompositeShape> RVOVisualizer::drawQuads(std::shared_ptr<CompositeShape> quadsRef) {
  std::shared_ptr<CompositeShape> quads;
  if(quadsRef)
    quads=quadsRef;
  else quads.reset(new CompositeShape);
  if(!_quadsUpdate)
    return quads;
  while(quads->numChildren()>0)
    quads->removeChild(quads->getChild(0));
  for(int i=0; i<(int)_qss.size(); i++) {
    std::shared_ptr<MeshShape> quad(new MeshShape);
    quad->addVertex(Eigen::Matrix<float,3,1>((float)std::get<0>(_qss[i])[0],(float)std::get<0>(_qss[i])[1],0));
    quad->addVertex(Eigen::Matrix<float,3,1>((float)std::get<1>(_qss[i])[0],(float)std::get<0>(_qss[i])[1],0));
    quad->addVertex(Eigen::Matrix<float,3,1>((float)std::get<1>(_qss[i])[0],(float)std::get<1>(_qss[i])[1],0));
    quad->addVertex(Eigen::Matrix<float,3,1>((float)std::get<0>(_qss[i])[0],(float)std::get<1>(_qss[i])[1],0));
    quad->addIndexSingle(0);
    quad->addIndexSingle(1);
    quad->addIndexSingle(2);
    quad->addIndexSingle(3);
    quad->setMode(GL_TRIANGLE_FAN);
    quad->setColorDiffuse(GL_TRIANGLE_FAN,std::get<2>(_qss[i])[0],std::get<2>(_qss[i])[1],std::get<2>(_qss[i])[2]);
    quads->addShape(quad);
  }
  _quadsUpdate=false;
  return quads;
}
void RVOVisualizer::drawVisibleApp(int argc,char** argv,bool offscreen,float ext,const RVOSimulator& sim,const std::vector<Eigen::Matrix<LSCALAR,2,1>>& vss,const std::vector<Eigen::Matrix<LSCALAR,2,1>>& nvss) {
  _drawer.reset(new Drawer(argc,argv));
  _exporter.reset(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat"));
  _capturer.reset(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",_drawer->FPS()));
  _drawer->addPlugin(_exporter);
  _drawer->addPlugin(_capturer);
  _drawer->addShape(drawRVOPosition(sim));
  if(!vss.empty())
    _drawer->addShape(drawLines(vss,Eigen::Matrix<float,3,1>(.7,.2,.2)));
  if(!nvss.empty())
    _drawer->addShape(drawLines(nvss,Eigen::Matrix<float,3,1>(.2,.7,.7)));
  _drawer->addCamera2D(ext);
  _drawer->clearLight();
  if(!offscreen)
    _drawer->mainLoop();
}
void RVOVisualizer::drawRVO(int argc,char** argv,bool offscreen,float ext,const RVOSimulator& sim,std::function<void()> frm,std::shared_ptr<RVOPythonCallback> cb) {
  _drawer.reset(new Drawer(argc,argv));
  if(cb)
    _drawer->setPythonCallback(cb.get());
  _exporter.reset(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat"));
  _capturer.reset(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",_drawer->FPS()));
  _drawer->addPlugin(_exporter);
  _drawer->addPlugin(_capturer);
  _agent=drawRVOPosition(sim);
  _vel=drawRVOVelocity(sim);
  _drawer->addShape(_lines=drawLines(_lines));
  _drawer->addShape(_quads=drawQuads(_quads));
  _drawer->addShape(_agent);
  _drawer->addCamera2D(ext);
  _drawer->clearLight();
  _frm=frm;
  bool step=false;
  _drawer->setKeyFunc([&](GLFWwindowPtr,int key,int,int action,int,bool captured) {
    if(captured)
      return;
    if(key==GLFW_KEY_R && action==GLFW_PRESS)
      step=!step;
    if(key==GLFW_KEY_W && action==GLFW_PRESS) {
      if(_drawer->contain(_vel))
        _drawer->removeShape(_vel);
      else _drawer->addShape(_vel);
    }
  });
  _drawer->setFrameFunc([&](std::shared_ptr<SceneNode>&) {
    if(step)
      _frm();
    drawLines(_lines);
    drawQuads(_quads);
    drawRVOPosition(sim,_agent);
    drawRVOVelocity(sim,_vel);
  });
  if(!offscreen) {
    _drawer->addPlugin(std::shared_ptr<Plugin>(new ImGuiPlugin([&]() {
      ImGui::Begin("Single-RVO Info");
      ImGui::Text("Simulation(r) %s",step?"started":"stopped");
      ImGui::Text("Velocity(w) %s",_drawer->contain(_vel)?"showing":"not showing");
      ImGui::End();
    })));
    _drawer->mainLoop();
  }
}
void RVOVisualizer::drawRVO(int argc,char** argv,bool offscreen,float ext,const MultiRVOSimulator& sim,std::function<void()> frm,std::shared_ptr<RVOPythonCallback> cb) {
  _drawer.reset(new Drawer(argc,argv));
  if(cb)
    _drawer->setPythonCallback(cb.get());
  _exporter.reset(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat"));
  _capturer.reset(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",_drawer->FPS()));
  _drawer->addPlugin(_exporter);
  _drawer->addPlugin(_capturer);
  _agent=drawRVOPosition(sim.getSubSimulator(0));
  _vel=drawRVOVelocity(sim.getSubSimulator(0));
  _drawer->addShape(_lines=drawLines(_lines));
  _drawer->addShape(_quads=drawQuads(_quads));
  _drawer->addShape(_agent);
  _drawer->addCamera2D(ext);
  _drawer->clearLight();
  _frm=frm;
  bool step=false;
  int id=0;
  _drawer->setKeyFunc([&](GLFWwindowPtr,int key,int,int action,int,bool captured) {
    if(captured)
      return;
    if(key==GLFW_KEY_R && action==GLFW_PRESS)
      step=!step;
    if(key==GLFW_KEY_D && action==GLFW_PRESS) {
      id=(id+1)%sim.getBatchSize();
      drawRVOPosition(sim.getSubSimulator(id),_agent);
      drawRVOVelocity(sim.getSubSimulator(id),_vel);
    }
    if(key==GLFW_KEY_A && action==GLFW_PRESS) {
      id=(id+sim.getBatchSize()-1)%sim.getBatchSize();
      drawRVOPosition(sim.getSubSimulator(id),_agent);
      drawRVOVelocity(sim.getSubSimulator(id),_vel);
    }
    if(key==GLFW_KEY_W && action==GLFW_PRESS) {
      if(_drawer->contain(_vel))
        _drawer->removeShape(_vel);
      else _drawer->addShape(_vel);
    }
  });
  _drawer->setFrameFunc([&](std::shared_ptr<SceneNode>&) {
    if(step)
      _frm();
    drawLines(_lines);
    drawQuads(_quads);
    drawRVOPosition(sim.getSubSimulator(id),_agent);
    drawRVOVelocity(sim.getSubSimulator(id),_vel);
  });
  if(!offscreen) {
    _drawer->addPlugin(std::shared_ptr<Plugin>(new ImGuiPlugin([&]() {
      ImGui::Begin("Multi-RVO Info");
      ImGui::Text("SimulatorID(ad): %d",id);
      ImGui::Text("Simulation(r) %s",step?"started":"stopped");
      ImGui::Text("Velocity(w) %s",_drawer->contain(_vel)?"showing":"not showing");
      ImGui::End();
    })));
    _drawer->mainLoop();
  }
}
void RVOVisualizer::drawRVO(int argc,char** argv,bool offscreen,float ext,const std::vector<Trajectory>& trajs,const RVOSimulator& sim,std::function<void()> frm,std::shared_ptr<RVOPythonCallback> cb) {
  _drawer.reset(new Drawer(argc,argv));
  if(cb)
    _drawer->setPythonCallback(cb.get());
  int frameId=0;
  _exporter.reset(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat"));
  _capturer.reset(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",_drawer->FPS()));
  _drawer->addPlugin(_exporter);
  _drawer->addPlugin(_capturer);
  _agent=drawRVOPosition(frameId,trajs,sim);
  _drawer->addShape(_lines=drawLines(_lines));
  _drawer->addShape(_quads=drawQuads(_quads));
  _drawer->addShape(_agent);
  _drawer->addCamera2D(ext);
  _drawer->clearLight();
  _frm=frm;
  bool step=false;
  _drawer->setKeyFunc([&](GLFWwindowPtr,int key,int,int action,int,bool captured) {
    if(captured)
      return;
    if(key==GLFW_KEY_R && action==GLFW_PRESS)
      step=!step;
    if(key==GLFW_KEY_W && action==GLFW_PRESS)
      frameId=0;
  });
  _drawer->setFrameFunc([&](std::shared_ptr<SceneNode>&) {
    if(step) {
      frameId++;
      _frm();
    }
    drawLines(_lines);
    drawQuads(_quads);
    drawRVOPosition(frameId,trajs,sim,_agent);
  });
  if(!offscreen) {
    _drawer->addPlugin(std::shared_ptr<Plugin>(new ImGuiPlugin([&]() {
      ImGui::Begin("Recorded Single-RVO Info");
      ImGui::Text("Replay(r) %s",step?"started":"stopped");
      ImGui::End();
    })));
    _drawer->mainLoop();
  }
}
void RVOVisualizer::drawRVO(int argc,char** argv,bool offscreen,float ext,const std::vector<std::vector<Trajectory>>& trajs,const MultiRVOSimulator& sim,std::function<void()> frm,std::shared_ptr<RVOPythonCallback> cb) {
  _drawer.reset(new Drawer(argc,argv));
  if(cb)
    _drawer->setPythonCallback(cb.get());
  int frameId=0;
  _exporter.reset(new CameraExportPlugin(GLFW_KEY_2,GLFW_KEY_3,"camera.dat"));
  _capturer.reset(new CaptureGIFPlugin(GLFW_KEY_1,"record.gif",_drawer->FPS()));
  _drawer->addPlugin(_exporter);
  _drawer->addPlugin(_capturer);
  _agent=drawRVOPosition(frameId,trajs[0],sim.getSubSimulator(0));
  _drawer->addShape(_lines=drawLines(_lines));
  _drawer->addShape(_quads=drawQuads(_quads));
  _drawer->addShape(_agent);
  _drawer->addCamera2D(ext);
  _drawer->clearLight();
  _frm=frm;
  bool step=false;
  int id=0;
  _drawer->setKeyFunc([&](GLFWwindowPtr,int key,int,int action,int,bool captured) {
    if(captured)
      return;
    if(key==GLFW_KEY_R && action==GLFW_PRESS)
      step=!step;
    if(key==GLFW_KEY_D && action==GLFW_PRESS) {
      id=(id+1)%(int)trajs.size();
      drawRVOPosition(frameId,trajs[id],sim.getSubSimulator(0),_agent);
    }
    if(key==GLFW_KEY_A && action==GLFW_PRESS) {
      id=(id+(int)trajs.size()-1)%(int)trajs.size();
      drawRVOPosition(frameId,trajs[id],sim.getSubSimulator(0),_agent);
    }
    if(key==GLFW_KEY_W && action==GLFW_PRESS)
      frameId=0;
  });
  _drawer->setFrameFunc([&](std::shared_ptr<SceneNode>&) {
    if(step) {
      frameId++;
      _frm();
    }
    drawLines(_lines);
    drawQuads(_quads);
    drawRVOPosition(frameId,trajs[id],sim.getSubSimulator(0),_agent);
  });
  if(!offscreen) {
    _drawer->addPlugin(std::shared_ptr<Plugin>(new ImGuiPlugin([&]() {
      ImGui::Begin("Recorded Multi-RVO Info");
      ImGui::Text("SimulatorID(ad): %d",id);
      ImGui::Text("Replay(r) %s",step?"started":"stopped");
      ImGui::End();
    })));
    _drawer->mainLoop();
  }
}
//convenient functions
void RVOVisualizer::drawRVO(bool offscreen,float ext,RVOSimulator& sim) {
  RVOVisualizer::drawRVO(0,NULL,offscreen,ext,sim,[&]() {
    sim.updateAgentTargets();
    sim.optimize(false,false);
  },NULL);
}
void RVOVisualizer::drawRVO(bool offscreen,float ext,MultiRVOSimulator& sim) {
  RVOVisualizer::drawRVO(0,NULL,offscreen,ext,sim,[&]() {
    sim.updateAgentTargets();
    sim.optimize(false,false);
  },NULL);
}
void RVOVisualizer::drawRVO(bool offscreen,float ext,RVOSimulator& sim,std::shared_ptr<RVOPythonCallback> cb) {
  RVOVisualizer::drawRVO(0,NULL,offscreen,ext,sim,[&]() {},cb);
}
void RVOVisualizer::drawRVO(bool offscreen,float ext,MultiRVOSimulator& sim,std::shared_ptr<RVOPythonCallback> cb) {
  RVOVisualizer::drawRVO(0,NULL,offscreen,ext,sim,[&]() {},cb);
}
void RVOVisualizer::drawRVO(bool offscreen,float ext,const std::vector<Trajectory>& trajs,const RVOSimulator& sim) {
  RVOVisualizer::drawRVO(0,NULL,offscreen,ext,trajs,sim,[&]() {});
}
void RVOVisualizer::drawRVO(bool offscreen,float ext,const std::vector<std::vector<Trajectory>>& trajs,const MultiRVOSimulator& sim) {
  RVOVisualizer::drawRVO(0,NULL,offscreen,ext,trajs,sim,[&]() {});
}
void RVOVisualizer::drawRVO(bool offscreen,float ext,const std::vector<Trajectory>& trajs,const RVOSimulator& sim,std::shared_ptr<RVOPythonCallback> cb) {
  RVOVisualizer::drawRVO(0,NULL,offscreen,ext,trajs,sim,[&]() {},cb);
}
void RVOVisualizer::drawRVO(bool offscreen,float ext,const std::vector<std::vector<Trajectory>>& trajs,const MultiRVOSimulator& sim,std::shared_ptr<RVOPythonCallback> cb) {
  RVOVisualizer::drawRVO(0,NULL,offscreen,ext,trajs,sim,[&]() {},cb);
}
void RVOVisualizer::getScreenshot(int& width,int& height,std::vector<unsigned char>& data) {
  if(_drawer && _capturer) {
    _drawer->frame();
    _drawer->draw();
    _capturer->getScreenshot(width,height,data);
  }
}
void RVOVisualizer::takeScreenshot() {
  if(_drawer && _capturer) {
    _drawer->frame();
    _drawer->draw();
    _capturer->takeScreenshot();
  }
}
}

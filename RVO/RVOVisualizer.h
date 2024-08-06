#ifndef RVO_VISUALIZER_H
#define RVO_VISUALIZER_H

#include "RVO.h"
#include "MultiRVO.h"
#include "Visibility.h"
#include <TinyVisualizer/Drawer.h>
#include <TinyVisualizer/CaptureGIFPlugin.h>
#include <TinyVisualizer/CameraExportPlugin.h>

namespace DRAWER {
class CompositeShape;
class MeshShape;
}

using namespace DRAWER;
namespace RVO {
class RVOPythonCallback : public PythonCallback {
 public:
  void mouse(int button,int action,int mods) override;
  void wheel(double xoffset,double yoffset) override;
  void motion(double x,double y) override;
  void key(int key,int scan,int action,int mods) override;
  void frame(std::shared_ptr<SceneNode>& root) override;
  void draw() override;
  void setup() override;
  std::function<void(int,int,int)> _mouse;
  std::function<void(double,double)> _wheel;
  std::function<void(double,double)> _motion;
  std::function<void(int,int,int,int)> _key;
  std::function<void()> _frame;
  std::function<void()> _draw;
  std::function<void()> _setup;
};
class RVOVisualizer {
 public:
  void clearSourceColor();
  void setSourceColor(unsigned short sid,Eigen::Matrix<float,3,1> color);
  void drawQuad(Eigen::Matrix<float,2,1> from,Eigen::Matrix<float,2,1> to,Eigen::Matrix<float,3,1> color);
  void drawLine(Eigen::Matrix<float,2,1> from,Eigen::Matrix<float,2,1> to,Eigen::Matrix<float,3,1> color);
  void drawVisibility(const VisibilityGraph& graph,const Eigen::Matrix<LSCALAR,2,1> p);
  void drawVisibility(const VisibilityGraph& graph,int id=-1);
  void clearQuad();
  void clearLine();
  int getNrQuads();
  void setNrQuads(int nr);
  int getNrLines();
  void setNrLines(int nr);
  void drawObstacle(const RVOSimulator& sim,std::shared_ptr<DRAWER::CompositeShape> shapesInput=NULL);
  std::shared_ptr<DRAWER::CompositeShape> drawRVOPosition(const RVOSimulator& sim,std::shared_ptr<DRAWER::CompositeShape> shapesInput=NULL);
  std::shared_ptr<DRAWER::CompositeShape> drawRVOPosition(int frameId,const std::vector<Trajectory>& trajectories,const RVOSimulator& sim,std::shared_ptr<CompositeShape> shapesInput=NULL);
  std::shared_ptr<DRAWER::MeshShape> drawRVOVelocity(const RVOSimulator& sim,std::shared_ptr<DRAWER::MeshShape> shapesInput=NULL);
  std::shared_ptr<MeshShape> drawLines(const std::vector<Eigen::Matrix<LSCALAR,2,1>>& vss,const Eigen::Matrix<float,3,1>& color);
  std::shared_ptr<CompositeShape> drawLines(std::shared_ptr<CompositeShape> linesRef);
  std::shared_ptr<CompositeShape> drawQuads(std::shared_ptr<CompositeShape> quadsRef);
  void drawVisibleApp(int argc,char** argv,bool offscreen,float ext,const RVOSimulator& sim,const std::vector<Eigen::Matrix<LSCALAR,2,1>>& vss,const std::vector<Eigen::Matrix<LSCALAR,2,1>>& nvss);
  void drawRVO(int argc,char** argv,bool offscreen,float ext,const RVOSimulator& sim,std::function<void()> frm,std::shared_ptr<RVOPythonCallback> cb=NULL);
  void drawRVO(int argc,char** argv,bool offscreen,float ext,const MultiRVOSimulator& sim,std::function<void()> frm,std::shared_ptr<RVOPythonCallback> cb=NULL);
  void drawRVO(int argc,char** argv,bool offscreen,float ext,const std::vector<Trajectory>& trajs,const RVOSimulator& sim,std::function<void()> frm,std::shared_ptr<RVOPythonCallback> cb=NULL);
  void drawRVO(int argc,char** argv,bool offscreen,float ext,const std::vector<std::vector<Trajectory>>& trajs,const MultiRVOSimulator& sim,std::function<void()> frm,std::shared_ptr<RVOPythonCallback> cb=NULL);
  //convenient functions
  void drawRVO(bool offscreen,float ext,RVOSimulator& sim);
  void drawRVO(bool offscreen,float ext,MultiRVOSimulator& sim);
  void drawRVO(bool offscreen,float ext,RVOSimulator& sim,std::shared_ptr<RVOPythonCallback> cb);
  void drawRVO(bool offscreen,float ext,MultiRVOSimulator& sim,std::shared_ptr<RVOPythonCallback> cb);
  void drawRVO(bool offscreen,float ext,const std::vector<Trajectory>& trajs,const RVOSimulator& sim);
  void drawRVO(bool offscreen,float ext,const std::vector<std::vector<Trajectory>>& trajs,const MultiRVOSimulator& sim);
  void drawRVO(bool offscreen,float ext,const std::vector<Trajectory>& trajs,const RVOSimulator& sim,std::shared_ptr<RVOPythonCallback> cb);
  void drawRVO(bool offscreen,float ext,const std::vector<std::vector<Trajectory>>& trajs,const MultiRVOSimulator& sim,std::shared_ptr<RVOPythonCallback> cb);
  void getScreenshot(int& width,int& height,std::vector<unsigned char>& data);
  void takeScreenshot();
 private:
  std::shared_ptr<Drawer> _drawer;
  std::shared_ptr<CompositeShape> _lines,_quads;
  std::shared_ptr<CameraExportPlugin> _exporter;
  std::shared_ptr<CaptureGIFPlugin> _capturer;
  std::shared_ptr<CompositeShape> _agent;
  std::shared_ptr<MeshShape> _vel;
  bool _quadsUpdate=true;
  bool _linesUpdate=true;
  std::function<void()> _frm=[]() {};
  std::unordered_map<unsigned short,Eigen::Matrix<float,3,1>> _css;
  std::vector<std::tuple<Eigen::Matrix<float,2,1>,Eigen::Matrix<float,2,1>,Eigen::Matrix<float,3,1>>> _qss;
  std::vector<std::tuple<Eigen::Matrix<float,2,1>,Eigen::Matrix<float,2,1>,Eigen::Matrix<float,3,1>>> _lss;
};
}

#endif

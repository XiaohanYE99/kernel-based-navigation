#ifndef RVO_VISUALIZER_H
#define RVO_VISUALIZER_H

#include "RVO.h"
#include "MultiRVO.h"
#include "Visibility.h"
#include <TinyVisualizer/Drawer.h>

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
  static void drawQuad(Eigen::Matrix<float,2,1> from,Eigen::Matrix<float,2,1> to,Eigen::Matrix<float,3,1> color);
  static void drawLine(Eigen::Matrix<float,2,1> from,Eigen::Matrix<float,2,1> to,Eigen::Matrix<float,3,1> color);
  static void drawVisibility(const VisibilityGraph& graph,const Eigen::Matrix<LSCALAR,2,1> p);
  static void drawVisibility(const VisibilityGraph& graph,int id=-1);
  static void clearQuad();
  static void clearLine();
  static int getNrQuads();
  static void setNrQuads(int nr);
  static int getNrLines();
  static void setNrLines(int nr);
  static void drawObstacle(const RVOSimulator& sim,std::shared_ptr<DRAWER::CompositeShape> shapesInput=NULL);
  static std::shared_ptr<DRAWER::CompositeShape> drawRVOPosition(const RVOSimulator& sim,std::shared_ptr<DRAWER::CompositeShape> shapesInput=NULL);
  static std::shared_ptr<DRAWER::CompositeShape> drawRVOPosition(int frameId,const std::vector<Trajectory>& trajectories,const RVOSimulator& sim,std::shared_ptr<CompositeShape> shapesInput=NULL);
  static std::shared_ptr<DRAWER::MeshShape> drawRVOVelocity(const RVOSimulator& sim,std::shared_ptr<DRAWER::MeshShape> shapesInput=NULL);
  static std::shared_ptr<MeshShape> drawLines(const std::vector<Eigen::Matrix<LSCALAR,2,1>>& vss,const Eigen::Matrix<float,3,1>& color);
  static std::shared_ptr<CompositeShape> drawLines(std::shared_ptr<CompositeShape> linesRef);
  static std::shared_ptr<CompositeShape> drawQuads(std::shared_ptr<CompositeShape> quadsRef);
  static void drawVisibleApp(int argc,char** argv,float ext,const RVOSimulator& sim,
                             const std::vector<Eigen::Matrix<LSCALAR,2,1>>& vss,
                             const std::vector<Eigen::Matrix<LSCALAR,2,1>>& nvss);
  static void drawRVO(int argc,char** argv,float ext,const RVOSimulator& sim,std::function<void()> frm,std::shared_ptr<RVOPythonCallback> cb=NULL);
  static void drawRVO(int argc,char** argv,float ext,const MultiRVOSimulator& sim,std::function<void()> frm,std::shared_ptr<RVOPythonCallback> cb=NULL);
  static void drawRVO(int argc,char** argv,float ext,const std::vector<Trajectory>& trajs,const RVOSimulator& sim,std::function<void()> frm,std::shared_ptr<RVOPythonCallback> cb=NULL);
  static void drawRVO(int argc,char** argv,float ext,const std::vector<std::vector<Trajectory>>& trajs,const MultiRVOSimulator& sim,std::function<void()> frm,std::shared_ptr<RVOPythonCallback> cb=NULL);
  //convenient functions
  static void drawRVO(float ext,RVOSimulator& sim);
  static void drawRVO(float ext,MultiRVOSimulator& sim);
  static void drawRVO(float ext,RVOSimulator& sim,std::shared_ptr<RVOPythonCallback> cb);
  static void drawRVO(float ext,MultiRVOSimulator& sim,std::shared_ptr<RVOPythonCallback> cb);
  static void drawRVO(float ext,const std::vector<Trajectory>& trajs,const RVOSimulator& sim);
  static void drawRVO(float ext,const std::vector<std::vector<Trajectory>>& trajs,const MultiRVOSimulator& sim);
  static void drawRVO(float ext,const std::vector<Trajectory>& trajs,const RVOSimulator& sim,std::shared_ptr<RVOPythonCallback> cb);
  static void drawRVO(float ext,const std::vector<std::vector<Trajectory>>& trajs,const MultiRVOSimulator& sim,std::shared_ptr<RVOPythonCallback> cb);
};
}

#endif

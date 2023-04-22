#ifndef RVO_VISUALIZER_H
#define RVO_VISUALIZER_H

#include "RVO.h"
#include "MultiRVO.h"
#include "Visibility.h"

namespace DRAWER {
class CompositeShape;
class PythonCallback;
class MeshShape;
}

using namespace DRAWER;
namespace RVO {
class RVOVisualizer {
 public:
  static void drawQuad(Eigen::Matrix<float,2,1> from,Eigen::Matrix<float,2,1> to,Eigen::Matrix<float,3,1> color);
  static void drawLine(Eigen::Matrix<float,2,1> from,Eigen::Matrix<float,2,1> to,Eigen::Matrix<float,3,1> color);
  static void drawVisibility(const VisibilityGraph& graph,const Eigen::Matrix<LSCALAR,2,1> p);
  static void drawVisibility(const VisibilityGraph& graph,int id=-1);
  static void clearQuad();
  static void clearLine();
#ifndef SWIG
  static std::shared_ptr<DRAWER::CompositeShape> drawRVOPosition(const RVOSimulator& sim,std::shared_ptr<DRAWER::CompositeShape> shapesInput=NULL);
  static std::shared_ptr<DRAWER::MeshShape> drawRVOVelocity(const RVOSimulator& sim,std::shared_ptr<DRAWER::MeshShape> shapesInput=NULL);
  static std::shared_ptr<MeshShape> drawLines(const std::vector<Eigen::Matrix<LSCALAR,2,1>>& vss,const Eigen::Matrix<float,3,1>& color);
  static std::shared_ptr<CompositeShape> drawLines();
  static std::shared_ptr<CompositeShape> drawQuads();
  static void drawVisibleApp(int argc,char** argv,float ext,const RVOSimulator& sim,
                             const std::vector<Eigen::Matrix<LSCALAR,2,1>>& vss,
                             const std::vector<Eigen::Matrix<LSCALAR,2,1>>& nvss);
  static void drawRVO(int argc,char** argv,float ext,const RVOSimulator& sim,std::function<void()> frm,PythonCallback* cb=NULL);
  static void drawRVO(int argc,char** argv,float ext,const MultiRVOSimulator& sim,std::function<void()> frm,PythonCallback* cb=NULL);
#endif
  static void drawRVO(float ext,RVOSimulator& sim);
  static void drawRVO(float ext,MultiRVOSimulator& sim);
  static void drawRVO(float ext,RVOSimulator& sim,PythonCallback* cb);
  static void drawRVO(float ext,MultiRVOSimulator& sim,PythonCallback* cb);
};
}

#endif

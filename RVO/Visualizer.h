#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "RVO.h"
#include "MultiRVO.h"

namespace DRAWER {
class CompositeShape;
class MeshShape;
}

namespace RVO {
#ifndef SWIG
extern std::shared_ptr<DRAWER::CompositeShape> drawRVOPosition(const RVOSimulator& sim,std::shared_ptr<DRAWER::CompositeShape> shapesInput=NULL);
extern std::shared_ptr<DRAWER::MeshShape> drawRVOVelocity(const RVOSimulator& sim,std::shared_ptr<DRAWER::MeshShape> shapesInput=NULL);
extern void drawVisibleApp(int argc,char** argv,float ext,const RVOSimulator& sim,
                           const std::vector<Eigen::Matrix<LSCALAR,2,1>>& vss,
                           const std::vector<Eigen::Matrix<LSCALAR,2,1>>& nvss);
extern void drawRVOApp(int argc,char** argv,float ext,const RVOSimulator& sim,std::function<void()> frm);
extern void drawRVOApp(int argc,char** argv,float ext,const MultiRVOSimulator& sim,std::function<void()> frm);
#endif
extern void drawRVOApp(float ext,RVOSimulator& sim);
extern void drawRVOApp(float ext,MultiRVOSimulator& sim);
}

#endif

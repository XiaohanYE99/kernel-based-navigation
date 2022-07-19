#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "RVO.h"
#include "MultiRVO.h"

namespace DRAWER {
class CompositeShape;
}

namespace RVO {
#ifndef SWIG
extern std::shared_ptr<DRAWER::CompositeShape> drawRVO(const RVOSimulator& sim,std::shared_ptr<DRAWER::CompositeShape> shapesInput=NULL);
extern void drawRVOApp(int argc,char** argv,float ext,const RVOSimulator& sim,std::function<void()> frm);
extern void drawRVOApp(int argc,char** argv,float ext,const MultiRVOSimulator& sim,std::function<void()> frm);
#endif
extern void drawRVOApp(float ext,RVOSimulator& sim);
extern void drawRVOApp(float ext,MultiRVOSimulator& sim);
}

#endif

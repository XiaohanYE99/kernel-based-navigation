#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "RVO.h"
namespace DRAWER {
class CompositeShape;
}

namespace RVO {
#ifndef SWIG
extern std::shared_ptr<DRAWER::CompositeShape> drawRVO(const RVOSimulator& sim,std::shared_ptr<DRAWER::CompositeShape> shapesInput=NULL);
extern void drawRVOApp(int argc,char** argv,float ext,const RVOSimulator& sim,std::function<void()> frm);
#endif
extern void drawRVOApp(float ext,RVOSimulator& sim);
}

#endif

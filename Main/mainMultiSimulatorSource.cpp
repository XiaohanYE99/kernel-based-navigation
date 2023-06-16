#define _USE_MATH_DEFINES
#include <cmath>
#include <RVO/MultiRVO.h>
#include <RVO/RVOVisualizer.h>
#include <chrono>

#define maxV 0.5
//#define CIRCLE
#define BLOCK
using namespace RVO;

int main(int argc,char** argv) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  T noise=5.;
  MultiRVOSimulator rvo(16,1,1e-4,1,1,1000,false,true,"NEWTON");
  rvo.setupSourceSink(1,10);
  rvo.addSourceSink(Vec2T(120,120),Vec2T(-120,-120),Vec2T(-130,-130),Vec2T(-110,-110),4,noise);
  rvo.addSourceSink(Vec2T(-120,-120),Vec2T(120,120),Vec2T(110,110),Vec2T(130,130),5,noise);
  rvo.addSourceSink(Vec2T(-120,120),Vec2T(120,-120),Vec2T(110,-130),Vec2T(130,-110),4,noise);
  rvo.addSourceSink(Vec2T(120,-120),Vec2T(-120,120),Vec2T(-130,110),Vec2T(-110,130),5,noise);
  rvo.addObstacle({Vec2T(-10,-10),Vec2T(10,-10),Vec2T(10,10),Vec2T(-10,10)});
  rvo.buildVisibility();
  //run
  for(int frameId=0; frameId<1000; frameId++) {
    rvo.updateAgentTargets();
    const auto beg=std::chrono::system_clock::now();
    rvo.optimize(false,false);
    std::cout << "frame=" << frameId << " cost=" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-beg).count() << "ms" << std::endl;
  }
  RVOVisualizer::drawRVO(argc,argv,150,rvo.getAllTrajectories(),rvo,[&]() {});
  return 0;
}

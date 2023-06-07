#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <RVO/RVO.h>
#include <RVO/SourceSink.h>
#include <RVO/RVOVisualizer.h>

#define maxV 0.5
//#define CIRCLE
//#define BLOCK
//#define DEBUG_BACKWARD
using namespace RVO;

int main(int argc,char** argv) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  SourceSink ss(1);
  ss.addSourceSink(Vec2T(120,120),Vec2T(-120,-120),BBox(Vec2T(-130,-130),Vec2T(-110,-110)),4);
  ss.addSourceSink(Vec2T(-120,-120),Vec2T(120,120),BBox(Vec2T(110,110),Vec2T(130,130)),5);
  ss.addSourceSink(Vec2T(-120,120),Vec2T(120,-120),BBox(Vec2T(110,-130),Vec2T(130,-110)),4);
  ss.addSourceSink(Vec2T(120,-120),Vec2T(-120,120),BBox(Vec2T(-130,110),Vec2T(-110,130)),5);
  RVOSimulator rvo(1,1e-4,1,1,1000,false,true,"NEWTON");
  rvo.addObstacle({Vec2T(-10,-10),Vec2T(10,-10),Vec2T(10,10),Vec2T(-10,10)});
  //run
  RVOVisualizer::drawRVO(argc,argv,150,rvo,[&]() {
    ss.addAgents(rvo);
    ss.removeAgents(rvo);
    rvo.updateAgentTargets();
    const auto beg=std::chrono::system_clock::now();
    rvo.optimize(false,false);
    std::cout << "cost=" << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now()-beg).count() << "s" << std::endl;
  });
  return 0;
}

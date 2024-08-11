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
  rvo.clearAgent();
  rvo.clearObstacle();
  rvo.clearSourceSink();
  rvo.clearVisibility();
  rvo.setupSourceSink(1,10,true);
  rvo.addSourceSink(Vec2T(120,120),Vec2T(-120,-120),Vec2T(-130,-130),Vec2T(-110,-110),4,noise);
  rvo.addSourceSink(Vec2T(-120,-120),Vec2T(120,120),Vec2T(110,110),Vec2T(130,130),5,noise);
  rvo.addSourceSink(Vec2T(-120,120),Vec2T(120,-120),Vec2T(110,-130),Vec2T(130,-110),4,noise);
  rvo.addSourceSink(Vec2T(120,-120),Vec2T(-120,120),Vec2T(-130,110),Vec2T(-110,130),5,noise);
  rvo.addObstacle({Vec2T(-10,-10),Vec2T(10,-10),Vec2T(10,10),Vec2T(-10,10)});
  rvo.buildVisibility();
  //run
  RVOVisualizer vis;
  vis.setSourceColor(0,Eigen::Matrix<float,3,1>(1,0,0));
  vis.setSourceColor(1,Eigen::Matrix<float,3,1>(0,1,0));
  vis.setSourceColor(2,Eigen::Matrix<float,3,1>(0,0,1));
  vis.setSourceColor(3,Eigen::Matrix<float,3,1>(1,0,1));
  vis.drawRVO(argc,argv,false,150,rvo,[&]() {
    rvo.updateAgentTargets();
    rvo.optimize(false,false);
  });
  return 0;
}

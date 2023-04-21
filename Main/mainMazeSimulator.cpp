#define _USE_MATH_DEFINES
#include <cmath>
#include <RVO/RVO.h>
#include <RVO/Visibility.h>
#include <RVO/RVOVisualizer.h>

#define maxV 0.5
using namespace RVO;

int main(int argc,char** argv) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  RVOSimulator rvo(1,1e-4,1,1,1000,false,true,"NEWTON");
  std::vector<T> oss({-170,-70,30,130});
  for(T x:oss)
    for(T y:oss) {
      Vec2T off(x,y);
      rvo.addObstacle({off,off+Vec2T(40,0),off+Vec2T(40,40),off+Vec2T(0,40)});
    }
  T rad;
  //bottom left
  rad=0.5;
  for(int x=-120; x<=-80; x+=10)
    for(int y=-120; y<=-80; y+=10) {
      int id=rvo.addAgent(Vec2T(x,y),Vec2T( 1, 1),rad);
      rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxV);
    }
  //top left
  rad=2;
  for(int x=-120; x<=-80; x+=10)
    for(int y=80; y<=120; y+=10) {
      int id=rvo.addAgent(Vec2T(x,y),Vec2T( 1,-1),rad);
      rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxV);
    }
  //bottom right
  rad=1;
  for(int x=80; x<=120; x+=10)
    for(int y=-120; y<=-80; y+=10) {
      int id=rvo.addAgent(Vec2T(x,y),Vec2T(-1, 1),rad);
      rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxV);
    }
  //top right
  rad=0.5;
  for(int x=80; x<=120; x+=10)
    for(int y=80; y<=120; y+=10) {
      int id=rvo.addAgent(Vec2T(x,y),Vec2T(-1,-1),0.5);
      rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxV);
    }
  //run
  VisibilityGraph graph(rvo);
  RVOVisualizer::drawVisibility(graph);
  RVOVisualizer::drawRVO(argc,argv,150,rvo,[&]() {
    rvo.updateAgentTargets();
    clock_t beg=clock();
    rvo.optimize(false,false);
    std::cout << "cost=" << (double)(clock()-beg)/CLOCKS_PER_SEC << "s" << std::endl;

  });
  return 0;
}

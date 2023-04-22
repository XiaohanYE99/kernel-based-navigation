#define _USE_MATH_DEFINES
#include <cmath>
#include <RVO/RVO.h>
#include <RVO/RVOVisualizer.h>

#define maxV 0.5
using namespace RVO;

int main(int argc,char** argv) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  RVOSimulator rvo(1,1e-4,1,1,1000,false,true,"NEWTON");
  rvo.addObstacle({Vec2T(-200,-20),Vec2T(200,-20),Vec2T(200,20),Vec2T(-200,20)});
  rvo.addObstacle({Vec2T(-20,-200),Vec2T(20,-200),Vec2T(20,200),Vec2T(-20,200)});
  rvo.buildVisibility();
  T rad;
  //bottom left
  rad=0.5;
  for(int x=-120; x<=-80; x+=10)
    for(int y=-120; y<=-80; y+=10) {
      int id=rvo.addAgent(Vec2T(x,y),Vec2T( 1, 1),rad);
      rvo.setAgentTarget(id,-rvo.getAgentPosition(id),1);
    }
  //top left
  rad=2;
  for(int x=-120; x<=-80; x+=10)
    for(int y=80; y<=120; y+=10) {
      int id=rvo.addAgent(Vec2T(x,y),Vec2T( 1,-1),rad);
      rvo.setAgentTarget(id,-rvo.getAgentPosition(id),1);
    }
  //bottom right
  rad=1;
  for(int x=80; x<=120; x+=10)
    for(int y=-120; y<=-80; y+=10) {
      int id=rvo.addAgent(Vec2T(x,y),Vec2T(-1, 1),rad);
      rvo.setAgentTarget(id,-rvo.getAgentPosition(id),1);
    }
  //top right
  rad=0.5;
  for(int x=80; x<=120; x+=10)
    for(int y=80; y<=120; y+=10) {
      int id=rvo.addAgent(Vec2T(x,y),Vec2T(-1,-1),0.5);
      rvo.setAgentTarget(id,-rvo.getAgentPosition(id),1);
    }
  //run
  RVOVisualizer::drawVisibility(*(rvo.getVisibility()));
  RVOVisualizer::drawRVO(argc,argv,150,rvo,[&]() {
    clock_t beg=clock();
    rvo.updateAgentTargets();
    rvo.optimize(false,false);
    std::cout << "cost=" << (double)(clock()-beg)/CLOCKS_PER_SEC << "s" << std::endl;
  });
  return 0;
}

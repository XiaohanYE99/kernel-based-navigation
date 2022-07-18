#include <RVO/RVO.h>
#include <RVO/Visualizer.h>
#include <thread>

#define maxV 0.1
#define CIRCLE
//#define BLOCK
using namespace RVO;

int main(int argc,char** argv) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  RVOSimulator rvo(2);
#ifdef CIRCLE
  for(const auto& off: {
        Vec2T(-50,-50),Vec2T(50,-50),Vec2T(50,50),Vec2T(-50,50)
      }) {
    std::vector<Vec2T> pos;
    for(int x=0; x<16; x++) {
      T theta=M_PI*2*x/16,r=20;
      pos.push_back(off+Vec2T(r*cos(theta),r*sin(theta)));
    }
    rvo.addObstacle(pos);
  }
#endif
#ifdef BLOCK
  for(const auto& off: {
        Vec2T(-70,-70),Vec2T(30,-70),Vec2T(30,30),Vec2T(-70,30)
      })
    rvo.addObstacle({off,off+Vec2T(40,0),off+Vec2T(40,40),off+Vec2T(0,40)});
#endif
  //bottom left
  for(int x=-120; x<=-80; x+=10)
    for(int y=-120; y<=-80; y+=10) {
      int id=rvo.addAgent(Vec2T(x,y),Vec2T( 1, 1));
      rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxV);
    }
  //top left
  for(int x=-120; x<=-80; x+=10)
    for(int y=80; y<=120; y+=10) {
      int id=rvo.addAgent(Vec2T(x,y),Vec2T( 1,-1));
      rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxV);
    }
  //bottom right
  for(int x=80; x<=120; x+=10)
    for(int y=-120; y<=-80; y+=10) {
      int id=rvo.addAgent(Vec2T(x,y),Vec2T(-1, 1));
      rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxV);
    }
  //top right
  for(int x=80; x<=120; x+=10)
    for(int y=80; y<=120; y+=10) {
      int id=rvo.addAgent(Vec2T(x,y),Vec2T(-1,-1));
      rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxV);
    }
  //run
  drawRVOApp(argc,argv,150,rvo,[&]() {
    rvo.optimize(NULL,NULL,true);
  });
  return 0;
}

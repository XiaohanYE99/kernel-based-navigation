#include <RVO/MultiRVO.h>
#include <RVO/Visualizer.h>

#define maxV 0.5
//#define CIRCLE
#define BLOCK
using namespace RVO;

int main(int argc,char** argv) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  MultiRVOSimulator rvo(5,2,1,1e-4,1,1,1000,false,true);
#ifdef CIRCLE
  for(const auto& off: {
        Vec2T(-50,-50),Vec2T(50,-50),Vec2T(50,50),Vec2T(-50,50)
      }) {
    std::vector<Vec2T> pos;
    for(int x=0; x<16; x++) {
      T theta=M_PI*2*x/16,r=20;
      pos.push_back(off+Vec2T(r*cos(theta),r*sin(theta)));
    }
    int id=rvo.addObstacle(pos);
    rvo.getObstacle(id);
  }
#endif
#ifdef BLOCK
  for(const auto& off: {
        Vec2T(-70,-70),Vec2T(30,-70),Vec2T(30,30),Vec2T(-70,30)
      }) {
    int id=rvo.addObstacle({off,off+Vec2T(40,0),off+Vec2T(40,40),off+Vec2T(0,40)});
    rvo.getObstacle(id);
  }
#endif
  T randRange=3;
  //bottom left
  for(int x=-120; x<=-80; x+=10)
    for(int y=-120; y<=-80; y+=10) {
      int id=rvo.addAgent({Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange},
      {Vec2T( 1, 1),Vec2T( 1, 1),Vec2T( 1, 1),Vec2T( 1, 1),Vec2T( 1, 1)});
      rvo.setAgentTarget(id, {-Vec2T(x,y),-Vec2T(x,y),-Vec2T(x,y),-Vec2T(x,y),-Vec2T(x,y)},maxV);
    }
  //top left
  for(int x=-120; x<=-80; x+=10)
    for(int y=80; y<=120; y+=10) {
      int id=rvo.addAgent({Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange},
      {Vec2T( 1,-1),Vec2T( 1,-1),Vec2T( 1,-1),Vec2T( 1,-1),Vec2T( 1,-1)});
      rvo.setAgentTarget(id, {-Vec2T(x,y),-Vec2T(x,y),-Vec2T(x,y),-Vec2T(x,y),-Vec2T(x,y)},maxV);
    }
  //bottom right
  for(int x=80; x<=120; x+=10)
    for(int y=-120; y<=-80; y+=10) {
      int id=rvo.addAgent({Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange},
      {Vec2T(-1, 1),Vec2T(-1, 1),Vec2T(-1, 1),Vec2T(-1, 1),Vec2T(-1, 1)});
      rvo.setAgentTarget(id, {-Vec2T(x,y),-Vec2T(x,y),-Vec2T(x,y),-Vec2T(x,y),-Vec2T(x,y)},maxV);
    }
  //top right
  for(int x=80; x<=120; x+=10)
    for(int y=80; y<=120; y+=10) {
      int id=rvo.addAgent({Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange,
                           Vec2T(x,y)+Vec2T::Random()*randRange},
      {Vec2T(-1,-1),Vec2T(-1,-1),Vec2T(-1,-1),Vec2T(-1,-1),Vec2T(-1,-1)});
      rvo.setAgentTarget(id, {-Vec2T(x,y),-Vec2T(x,y),-Vec2T(x,y),-Vec2T(x,y),-Vec2T(x,y)},maxV);
    }
  //run
  drawRVOApp(argc,argv,150,rvo,[&]() {
    rvo.updateAgentTargets();
    rvo.optimize(true,true);
    rvo.getDXDV();
    rvo.getDXDX();
  });
  return 0;
}

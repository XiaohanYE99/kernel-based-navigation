#define _USE_MATH_DEFINES
#include <cmath>
#include <RVO/RVO.h>
#include <RVO/RVOVisualizer.h>

#define maxV 0.5
//#define CIRCLE
//#define BLOCK
//#define DEBUG_BACKWARD
using namespace RVO;

int main(int argc,char** argv) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  RVOSimulator rvo(1,1e-4,1,1,1000,false,true,"NEWTON");
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
  T rad;
  /*rad=0.5;
  int id=rvo.addAgent(Vec2T(-50,-51),Vec2T( 1, 1),rad);
  rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxV);
  rad=1;
  id=rvo.addAgent(Vec2T(51,50),Vec2T( 1, 1),rad);
  rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxV);
  rad=1;
  id=rvo.addAgent(Vec2T(-50,55),Vec2T( 1, 1),rad);
  rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxV);
  rad=0.5;
  id=rvo.addAgent(Vec2T(52,-50),Vec2T( 1, 1),rad);
  rvo.setAgentTarget(id,-rvo.getAgentPosition(id),maxV);*/
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
  RVOVisualizer::drawLine({-50,0}, {50,0}, {1,0,0});
  RVOVisualizer::drawLine({0,-50}, {0,50}, {0,1,0});
  RVOVisualizer::drawQuad({-20,-20}, {20,20}, {.6,.6,.6});
  RVOVisualizer::drawRVO(argc,argv,150,rvo,[&]() {
    rvo.updateAgentTargets();
#ifdef DEBUG_BACKWARD
    Mat2XT pos=rvo.getAgentPositions();
    Mat2XT vel=rvo.getAgentVelocities();
#endif
    clock_t beg=clock();
    rvo.optimize(false,false);
    std::cout << "cost=" << (double)(clock()-beg)/CLOCKS_PER_SEC << "s" << std::endl;
#ifdef DEBUG_BACKWARD
    Mat2XT newPos=rvo.getAgentPositions();
    MatT DXDX=rvo.getDXDX(),DXDV=rvo.getDXDV();
    Mat2XT dx=Mat2XT::Random(pos.rows(),pos.cols());
    {
      T Delta=1e-4;
      rvo.getAgentPositions()=pos+dx*Delta;
      rvo.getAgentVelocities()=vel;
      rvo.optimize(false,true);
      Mat2XT newPos2=rvo.getAgentPositions();
      Eigen::Map<const Vec> dxM(dx.data(),dx.size());
      Eigen::Map<const Vec> newPosM(newPos.data(),newPos.size());
      Eigen::Map<const Vec> newPos2M(newPos2.data(),newPos2.size());
      DEBUG_GRADIENT("DXDX",(DXDX*dxM).norm(),(DXDX*dxM-(newPos2M-newPosM)/Delta).norm())
    }
    {
      T Delta=1e-4;
      rvo.getAgentPositions()=pos;
      rvo.getAgentVelocities()=vel+dx*Delta;
      rvo.optimize(false,true);
      Mat2XT newPos2=rvo.getAgentPositions();
      Eigen::Map<const Vec> dxM(dx.data(),dx.size());
      Eigen::Map<const Vec> newPosM(newPos.data(),newPos.size());
      Eigen::Map<const Vec> newPos2M(newPos2.data(),newPos2.size());
      DEBUG_GRADIENT("DXDV",(DXDV*dxM).norm(),(DXDV*dxM-(newPos2M-newPosM)/Delta).norm())
    }
#endif
  });
  return 0;
}

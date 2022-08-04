#include <RVO/RVO.h>
#include <RVO/ORCA.h>
#include <RVO/Visualizer.h>

#define maxV 0.5
//#define CIRCLE
//#define BLOCK
#define USE_ORCA
using namespace RVO;

int main(int argc,char** argv) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
#ifdef USE_ORCA
  ORCASimulator rvo(4,1,1e-4,1,1,1000,false,true);
#else
  RVOSimulator rvo(4,1,1e-4,1,1,1000,false,true);
#endif
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
    rvo.updateAgentTargets();
#ifdef DEBUG_BACKWARD
    Mat2XT pos=rvo.getAgentPositions();
    Mat2XT vel=rvo.getAgentVelocities();
#endif
    clock_t beg=clock();
    rvo.optimize(true,false);
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

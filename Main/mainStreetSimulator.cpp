#define _USE_MATH_DEFINES
#include <cmath>
#include <RVO/RVO.h>
#include <RVO/Visualizer.h>

using namespace RVO;

int main(int argc,char** argv) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  RVOSimulator rvo(1,1e-4,1,1,10,false,true,"NEWTON");
  //outer
  for(int x=0; x<32; x++) {
    T theta=M_PI*2*x/32,r=50,dr=1;
    T thetaNext=M_PI*2*(x+1)/32;
    std::vector<Vec2T> pos;
    pos.push_back(Vec2T(r*cos(theta),r*sin(theta)));
    pos.push_back(Vec2T((r+dr)*cos(theta),(r+dr)*sin(theta)));
    pos.push_back(Vec2T((r+dr)*cos(thetaNext),(r+dr)*sin(thetaNext)));
    pos.push_back(Vec2T(r*cos(thetaNext),r*sin(thetaNext)));
    rvo.addObstacle(pos);
  }
  //inner
  for(int x=0; x<32; x++) {
    T theta=M_PI*2*x/32,r=25,dr=1;
    T thetaNext=M_PI*2*(x+1)/32;
    std::vector<Vec2T> pos;
    pos.push_back(Vec2T(r*cos(theta),r*sin(theta)));
    pos.push_back(Vec2T((r+dr)*cos(theta),(r+dr)*sin(theta)));
    pos.push_back(Vec2T((r+dr)*cos(thetaNext),(r+dr)*sin(thetaNext)));
    pos.push_back(Vec2T(r*cos(thetaNext),r*sin(thetaNext)));
    rvo.addObstacle(pos);
  }
  T r=37;
  //add top group
  T rad=1;
  for(int x=-10; x<=10; x+=3)
    for(int y=-10; y<=10; y+=3)
      rvo.addAgent(Vec2T(x,y+r),Vec2T( 1, 1),rad);
  //add bottom group
  rad=0.5;
  for(int x=-10; x<=10; x+=3)
    for(int y=-10; y<=10; y+=3)
      rvo.addAgent(Vec2T(x,y-r),Vec2T( 1, 1),rad);
  //run
  bool output=true;
  std::vector<T> rss;
  rss.resize(rvo.getNrAgent());
  drawRVOApp(argc,argv,150,rvo,[&]() {
    rvo.updateAgentTargets();
    clock_t beg=clock();
    //set target
    int nGroup=rvo.getNrAgent()/2;
    for(int i=0; i<nGroup*2; i++) {
      Vec2T pos=rvo.getAgentPosition(i);
      rss[i]=pos.norm();
      T theta=atan2(pos[1],pos[0]),dtheta=i<nGroup?M_PI/60:-M_PI/60;
      rvo.setAgentTarget(i,Vec2T(rss[i]*cos(theta+dtheta),rss[i]*sin(theta+dtheta)),0.5);
    }
    //optimize
    rvo.optimize(false,output);
    std::cout << "cost=" << (double)(clock()-beg)/CLOCKS_PER_SEC << "s" << std::endl;
  });
  return 0;
}

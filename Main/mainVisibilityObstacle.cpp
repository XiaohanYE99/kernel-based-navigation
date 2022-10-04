#include <RVO/RVO.h>
#include <RVO/Visualizer.h>

using namespace RVO;

int main(int argc,char** argv) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  T scale=100;

  RVOSimulator rvo(100);
  for(int i=0; i<100; i++)
    rvo.addAgent(Vec2T::Random()*scale,Vec2T::Random()*scale,i<50?0.5:1);
  for(const auto& off: {
        Vec2T(-60,-60),Vec2T(40,-60),Vec2T(40,40),Vec2T(-60,40)
      })
    rvo.addObstacle({off,off+Vec2T(20,0),off+Vec2T(20,20),off+Vec2T(0,20)});

  //visibility
  std::vector<Vec2T> vss,nvss;
  for(int i=0; i<100; i++) {
    Vec2T a=Vec2T::Random()*scale,b;
    std::shared_ptr<Obstacle> o=rvo.getBVH().getVertex(rand()%rvo.getBVH().getNrVertex());
    if(rvo.getBVH().visible(a,o,&b)) {
      vss.push_back(a);
      vss.push_back(b);
    } else {
      nvss.push_back(a);
      nvss.push_back(b);
    }
  }
  drawVisibleApp(argc,argv,100,rvo,vss,nvss);
  return 0;
}

#include <RVO/RVO.h>
#include <RVO/CoverageEnergy.h>

using namespace RVO;

int main() {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  T scale=100;

  RVOSimulator rvo(1,100);
  for(int i=0; i<100; i++)
    rvo.addAgent(Vec2T::Random()*scale,Vec2T::Random()*scale);
  for(const auto& off: {
        Vec2T(-60,-60),Vec2T(40,-60),Vec2T(40,40),Vec2T(-60,40)
      })
    rvo.addObstacle({off,off+Vec2T(20,0),off+Vec2T(20,20),off+Vec2T(0,20)});
  //debug
  CoverageEnergy(rvo,50,true).debugCoverage(scale);
  CoverageEnergy(rvo,50,false).debugCoverage(scale);
  rvo.debugNeighbor(scale);
  rvo.debugEnergy(scale);
  return 0;
}

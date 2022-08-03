#include <RVO/ORCA.h>

using namespace RVO;

int main() {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  ORCASimulator orca(0.4);
  orca.setTimestep(0.9);
  for(int i=0; i<100; i++)
    orca.addAgent(Vec2T::Random(),Vec2T::Random());
  for(int iter=0; iter<100; iter++) {
    //agent-agent
    orca.debugVO(-1,-1,0);
    orca.debugVO(-1,-1,1);
    orca.debugVO(-1,-1,2);
    //agent-obstacle
    for(auto offset: {
          0,5
        }) {
      orca.debugVO(-1,0+offset,true);
      orca.debugVO(-1,1+offset,true);
      orca.debugVO(-1,2+offset,false);
      orca.debugVO(-1,3+offset,false);
      orca.debugVO(-1,4+offset,false);
    }
    //agent-obstacle
    orca.debugVO(-1,10,true);
    orca.debugVO(-1,11,true);
    orca.debugVO(-1,12,true);
    orca.debugVO(-1,13,false);
    orca.debugVO(-1,14,false);
    orca.debugVO(-1,15,false);
    orca.debugVO(-1,16,false);
    orca.debugVO(-1,17,false);
  }
  return 0;
}

#define _USE_MATH_DEFINES
#include <cmath>
#include <RVO/RVO.h>
#include <RVO/RVOVisualizer.h>
#include <TinyVisualizer/Drawer.h>

#define maxV 0.5
using namespace RVO;
using namespace DRAWER;
class VisibilityCallback : public PythonCallback {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  VisibilityCallback(RVOSimulator& rvo,T speed=5):_rvo(rvo),_speed(speed) {
    RVOVisualizer::drawVisibility(*(rvo.getVisibility()));
    _nr=RVOVisualizer::getNrLines();
    _src=Vec2T(0,0);
  }
  void key(int key,int scan,int action,int mods) override {
    bool change=false;
    if(key==GLFW_KEY_W && action==GLFW_PRESS) {
      _src.y()+=_speed;
      change=true;
    } else if(key==GLFW_KEY_S && action==GLFW_PRESS) {
      _src.y()-=_speed;
      change=true;
    } else if(key==GLFW_KEY_D && action==GLFW_PRESS) {
      _src.x()+=_speed;
      change=true;
    } else if(key==GLFW_KEY_A && action==GLFW_PRESS) {
      _src.x()-=_speed;
      change=true;
    }
    if(!change)
      return;
    RVOVisualizer::setNrLines(_nr);
    RVOVisualizer::drawVisibility(*(_rvo.getVisibility()),_src);
  }
  RVOSimulator& _rvo;
  Vec2T _src;
  T _speed;
  int _nr;
};
int main(int argc,char** argv) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  RVOSimulator rvo(1,1e-4,1,1,1000,false,true,"NEWTON");
  Eigen::Matrix<bool,-1,-1> maze;
  int sz=20;
  maze.resize(5,7);
  maze << 0,0,0,0,0,0,0,
       0,1,0,1,1,1,0,
       0,0,0,0,1,0,0,
       0,1,0,1,1,1,0,
       0,1,0,0,1,0,0;
  for(int r=-1; r<=maze.rows(); r++)
    for(int c=-1; c<=maze.cols(); c++)
      if(r<0 || r>=maze.rows() || c<0 || c>=maze.cols() || maze(r,c)) {
        Vec2T off(c*sz,(maze.rows()-1-r)*sz);
        rvo.addObstacle({Vec2T(0,0)+off,Vec2T(sz,0)+off,Vec2T(sz,sz)+off,Vec2T(0,sz)+off});
      }
  rvo.buildVisibility();
  //run
  VisibilityCallback cb(rvo);
  RVOVisualizer::drawRVO(argc,argv,150,rvo,[]() {},&cb);
  return 0;
}

#define _USE_MATH_DEFINES
#include <cmath>
#include <RVO/RVO.h>
#include <RVO/RVOVisualizer.h>
#include <TinyVisualizer/Drawer.h>

#define CASEI
using namespace RVO;
using namespace DRAWER;
int main(int argc,char** argv) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  RVOSimulator rvo(1,1e-4,1,1,1000,false,true,"NEWTON");
  Eigen::Matrix<bool,-1,-1> maze;
  int sz=20;
  maze.resize(5,7);
#ifdef CASEI
  maze <<
       0,0,0,0,0,0,0,
       0,1,0,1,1,1,0,
       0,0,0,0,1,0,0,
       0,1,0,1,1,1,0,
       0,1,0,0,1,0,0;
#else
  maze <<
       0,0,0,0,0,0,0,
       0,0,0,1,0,0,0,
       0,1,1,1,1,1,0,
       0,0,0,1,0,0,0,
       0,0,0,0,0,0,0;
#endif
  for(int r=-1; r<=maze.rows(); r++)
    for(int c=-1; c<=maze.cols(); c++)
      if(r<0 || r>=maze.rows() || c<0 || c>=maze.cols() || maze(r,c)) {
        Vec2T off(c*sz,(maze.rows()-1-r)*sz);
        rvo.addObstacle({Vec2T(0,0)+off,Vec2T(sz,0)+off,Vec2T(sz,sz)+off,Vec2T(0,sz)+off});
      }
  rvo.buildVisibility();
  //run
  RVOVisualizer::drawVisibility(*(rvo.getVisibility()));
  int nr=RVOVisualizer::getNrLines();
  Vec2T src=Vec2T(0,0);
  T speed=5;
  std::shared_ptr<RVOPythonCallback> cb(new RVOPythonCallback);
  auto key=[&](int key,int scan,int action,int mods)->void {
    bool change=false;
    if(key==GLFW_KEY_UP && action==GLFW_PRESS) {
      src.y()+=speed;
      change=true;
    } else if(key==GLFW_KEY_DOWN && action==GLFW_PRESS) {
      src.y()-=speed;
      change=true;
    } else if(key==GLFW_KEY_RIGHT && action==GLFW_PRESS) {
      src.x()+=speed;
      change=true;
    } else if(key==GLFW_KEY_LEFT && action==GLFW_PRESS) {
      src.x()-=speed;
      change=true;
    }
    if(!change)
      return;
    RVOVisualizer::setNrLines(nr);
    RVOVisualizer::drawVisibility(*(rvo.getVisibility()),src);
    std::cout << "Checking visibility at: " << src.transpose() << std::endl;
  };
  cb->_key=key;
  RVOVisualizer::drawRVO(argc,argv,150,rvo,[]() {},cb);
  return 0;
}

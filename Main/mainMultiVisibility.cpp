#define _USE_MATH_DEFINES
#include <cmath>
#include <RVO/MultiRVO.h>
#include <RVO/RVOVisualizer.h>
#include <RVO/MultiVisibility.h>
#include <chrono>

#define maxV 0.5
//#define CIRCLE
#define BLOCK
using namespace RVO;
typedef LSCALAR T;
DECL_MAT_VEC_MAP_TYPES_T

std::vector<Vec2T> mat2vec(const Mat2XT& mat) {
  std::vector<Vec2T> ret;
  ret.resize(mat.cols());
  for(int i=0; i<(int)ret.size(); i++)
    ret[i]=mat.col(i);
  return ret;
}
Mat2XT vec2mat(const std::vector<Vec2T>& vec) {
  Mat2XT ret;
  ret.resize(2,vec.size());
  for(int i=0; i<(int)ret.cols(); i++)
    ret.col(i)=vec[i];
  return ret;
}
int main(int argc,char** argv) {
  T noise=5.;
  int batchId=1;
  MultiRVOSimulator rvo(batchId,1,1e-4,1,1,1000,false,true,"NEWTON");
  rvo.clearAgent();
  rvo.clearObstacle();
  rvo.clearSourceSink();
  rvo.clearVisibility();
  rvo.setupSourceSink(1,10,true);
  rvo.addSourceSink(Vec2T(120,120),Vec2T(-120,-120),Vec2T(-130,-130),Vec2T(-110,-110),4,noise);
  rvo.addSourceSink(Vec2T(-120,-120),Vec2T(120,120),Vec2T(110,110),Vec2T(130,130),5,noise);
  rvo.addSourceSink(Vec2T(-120,120),Vec2T(120,-120),Vec2T(110,-130),Vec2T(130,-110),4,noise);
  rvo.addSourceSink(Vec2T(120,-120),Vec2T(-120,120),Vec2T(-130,110),Vec2T(-110,130),5,noise);
  rvo.addObstacle({Vec2T(-10,-10),Vec2T(0,-15),Vec2T(10,-10),Vec2T(15,0),Vec2T(10,10),Vec2T(0,15),Vec2T(-10,10),Vec2T(-15,0)});
  MultiVisibilityGraph vis(rvo);
  //run
  for(int frameId=0; frameId<500; frameId++) {
    for(int id=0; id<batchId; id++) {
      Mat2XT tss=rvo.getAllAgentTargets(id);
      Mat2XT pss=rvo.getAllAgentPositions(id);
      vis.setAgentTargets(mat2vec(tss),1);
      std::vector<Vec2T> vss=vis.setAgentPositions(mat2vec(pss));
      std::vector<T> distance=vis.getMinDistance();
      std::cout << "Batch" << id << " distance:";
      for(int i=0; i<(int)distance.size(); i++)
        std::cout << " " << distance[i];
      std::cout << std::endl;
      rvo.setAllAgentVelocities(id,vec2mat(vss));
    }
    const auto beg=std::chrono::system_clock::now();
    rvo.optimize(false,false);
    std::cout << "frame=" << frameId << " cost=" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-beg).count() << "ms" << std::endl;
  }
  RVOVisualizer::drawRVO(argc,argv,150,rvo.getAllTrajectories(),rvo,[&]() {});
  return 0;
}

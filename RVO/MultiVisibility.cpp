#include "MultiVisibility.h"

namespace RVO {
//MultiVisibility
MultiVisibilityGraph::MultiVisibilityGraph(RVOSimulator& rvo):VisibilityGraph(rvo) {}
MultiVisibilityGraph::MultiVisibilityGraph(MultiRVOSimulator& rvo):VisibilityGraph(rvo.getSubSimulator(0)) {
  if(rvo.hasSourceSink()) {
    const SourceSink& ss=rvo.getSubSourceSink(0);
    const DynamicMat<T>& targetPos=ss.getTargetPos();
    for(int c=0; c<targetPos.cols(); c++) {
      Vec2T t=targetPos.getCMap().col(c);
      _pathCache[t]=buildShortestPath(t);
    }
    std::cout << "MultiVisibilityGraph using pathCache!" << std::endl;
  } else {
    std::cout << "MultiVisibilityGraph not using pathCache!" << std::endl;
  }
}
void MultiVisibilityGraph::setAgentTargets(const std::vector<Vec2T>& target,T maxVelocity) {
  _pathVec.resize(target.size());
  if(_pathCache.empty()) {
    OMP_PARALLEL_FOR_
    for(int i=0; i<(int)_pathVec.size(); i++) {
      _pathVec[i]=buildShortestPath(target[i]);
      _pathVec[i]._maxVelocity=maxVelocity;
    }
  } else {
    OMP_PARALLEL_FOR_
    for(int i=0; i<(int)_pathVec.size(); i++) {
      auto it=_pathCache.find(target[i]);
      ASSERT_MSG(it!=_pathCache.end(),"Cannot find path in pathCache!")
      _pathVec[i]=it->second;
      _pathVec[i]._maxVelocity=maxVelocity;
    }
  }
}
std::vector<MultiVisibilityGraph::Vec2T> MultiVisibilityGraph::setAgentPositions(const std::vector<Vec2T>& positions) {
  std::vector<Vec2T> ret(_pathVec.size());
  _minDistance.resize(_pathVec.size());
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_pathVec.size(); i++) {
    Vec2T dir=getAgentWayPoint(_pathVec[i],positions[i],_minDistance[i])-positions[i];
    T len=dir.norm();
    if(len>_pathVec[i]._maxVelocity) {
      T coef=_pathVec[i]._maxVelocity/len;
      ret[i]=dir*coef;
      _pathVec[i]._DVDP=(ret[i]*ret[i].transpose()-Mat2T::Identity())*coef;
    } else {
      ret[i]=dir;
      _pathVec[i]._DVDP=-Mat2T::Identity();
    }
  }
  return ret;
}
std::vector<MultiVisibilityGraph::Mat2T> MultiVisibilityGraph::getAgentDVDPs() const {
  std::vector<Mat2T> ret(_pathVec.size());
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_pathVec.size(); i++)
    ret[i]=_pathVec[i]._DVDP;
  return ret;
}
std::vector<MultiVisibilityGraph::T> MultiVisibilityGraph::getMinDistance() const {
  return _minDistance;
}
}

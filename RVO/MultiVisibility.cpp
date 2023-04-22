#include "MultiVisibility.h"

namespace RVO {
//MultiVisibility
MultiVisibilityGraph::MultiVisibilityGraph(RVOSimulator& rvo):VisibilityGraph(rvo) {}
MultiVisibilityGraph::MultiVisibilityGraph(MultiRVOSimulator& rvo):VisibilityGraph(rvo.getSubSimulator(0)) {}
void MultiVisibilityGraph::setAgentTargets(const std::vector<Vec2T>& target,T maxVelocity) {
  _pathVec.resize(target.size());
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_pathVec.size(); i++) {
    _pathVec[i]=buildShortestPath(target[i]);
    _pathVec[i]._maxVelocity=maxVelocity;
  }
}
std::vector<MultiVisibilityGraph::Vec2T> MultiVisibilityGraph::setAgentPositions(const std::vector<Vec2T>& positions) {
  std::vector<Vec2T> ret(_pathVec.size());
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_pathVec.size(); i++) {
    Vec2T dir=getAgentWayPoint(_pathVec[i],positions[i])-positions[i];
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
}

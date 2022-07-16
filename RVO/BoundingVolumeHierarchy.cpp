#include "BoundingVolumeHierarchy.h"

namespace RVO {
void BoundingVolumeHierarchy::clearObstacle() {
  _obs.clear();
  _bvh.clear();
}
void BoundingVolumeHierarchy::addObstacle(const std::vector<Vec2T>& vss) {
  int offset=(int)_obs.size();
  for(int i=0; i<(int)vss.size(); i++)
    _obs.push_back(std::shared_ptr<Obstacle>(new Obstacle(vss[i])));
  for(int i=0; i<(int)vss.size(); i++)
    _obs[offset+i]->_next=_obs[offset+(i+1)%(int)vss.size()];
  assemble();
}
const std::vector<std::shared_ptr<Obstacle>>& BoundingVolumeHierarchy::getObstacles() const {
  return _obs;
}
const std::vector<Node<int,BBox>>& BoundingVolumeHierarchy::getNodes() const {
  return _bvh;
}
void BoundingVolumeHierarchy::assemble() {
  _bvh.clear();
  std::unordered_set<Eigen::Matrix<int,2,1>,EdgeHash<int>> edgeMap;
  for(int i=0; i<(int)_obs.size(); i++) {
    _bvh.push_back(Node<int,BBox>());
    _bvh.back()._nrCell=1;
    _bvh.back()._cell=i;
    _bvh.back()._bb.setUnion(_obs[i]->_pos);
    _bvh.back()._bb.setUnion(_obs[i]->_next->_pos);
    //consecutive edges
    if(i<(int)_obs.size()-1 && _obs[i]->_next==_obs[i+1])
      edgeMap.insert(Eigen::Matrix<int,2,1>(i,i+1));
  }
  Node<int,BBox>::buildBVHBottomUp(_bvh,edgeMap,false);
}
}

#include "BoundingVolumeHierarchy.h"
#include <stack>

namespace RVO {
void BoundingVolumeHierarchy::clearObstacle() {
  _obs.clear();
  _bvh.clear();
}
int BoundingVolumeHierarchy::getNrObstacle() const {
  int nr=0;
  for(int id=0; id<(int)_obs.size(); id++)
    if(_obs[id]->_next->_id<id)
      nr++;
  return nr;
}
std::vector<BoundingVolumeHierarchy::Vec2T> BoundingVolumeHierarchy::getObstacle(int i) const {
  int nr=0;
  std::vector<Vec2T> pos;
  for(int id=0; id<(int)_obs.size(); id++)
    if(_obs[id]->_next->_id<id) {
      if(nr==i)
        for(int idBeg=_obs[id]->_next->_id; idBeg<=id; idBeg++)
          pos.push_back(_obs[idBeg]->_pos);
      nr++;
    }
  return pos;
}
int BoundingVolumeHierarchy::getNrVertex() const {
  return (int)_obs.size();
}
std::shared_ptr<Obstacle> BoundingVolumeHierarchy::getVertex(int i) const {
  return _obs[i];
}
void BoundingVolumeHierarchy::addObstacle(const std::vector<Vec2T>& vss) {
  int offset=(int)_obs.size();
  for(int i=0; i<(int)vss.size(); i++)
    _obs.push_back(std::shared_ptr<Obstacle>(new Obstacle(vss[i],(int)_obs.size())));
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
bool BoundingVolumeHierarchy::visible(const Vec2T& a,const Vec2T& b,std::shared_ptr<Obstacle> obs) const {
  BBox bb;
  bb.setUnion(a);
  bb.setUnion(b);
  Vec2T edgeA[2]= {a,b};
  //stack
  std::stack<int> ss;
  ss.push((int)_bvh.size()-1);
  while(!ss.empty()) {
    int curr=ss.top();
    ss.pop();
    if(!_bvh[curr]._bb.intersect(bb))
      continue;
    else if(_bvh[curr]._cell>=0) {
      if(obs) {
        if(_obs[_bvh[curr]._cell]==obs || _obs[_bvh[curr]._cell]->_next==obs)
          continue;
        if(_obs[_bvh[curr]._cell]==obs->_next || _obs[_bvh[curr]._cell]->_next==obs->_next)
          continue;
      }
      Vec2T edgeB[2]= {_obs[_bvh[curr]._cell]->_pos,_obs[_bvh[curr]._cell]->_next->_pos};
      if(intersect(edgeA,edgeB))
        return false;
    } else {
      ss.push(_bvh[curr]._l);
      ss.push(_bvh[curr]._r);
    }
  }
  return true;
}
bool BoundingVolumeHierarchy::visible(const Vec2T& a,std::shared_ptr<Obstacle> obs,Vec2T* bRef) const {
  Vec2T o[2]= {obs->_pos,obs->_next->_pos};
  Vec2T obsVec=o[1]-o[0],relPos0=o[0]-a;
  T lenSq=obsVec.squaredNorm(),s=(-relPos0.dot(obsVec))/lenSq;
  s=fmin((T)1.,fmax((T)0.,s));
  Vec2T b=o[0]*(1-s)+o[1]*s;
  if(bRef)
    *bRef=b;
  return visible(a,b,obs);
}
bool BoundingVolumeHierarchy::intersect(const Vec2T edgeA[2],const Vec2T edgeB[2]) {
  //edgeA[0]+s*(edgeA[1]-edgeA[0])=edgeB[0]+t*(edgeB[1]-edgeB[0])
  Mat2T LHS;
  Vec2T RHS=edgeB[0]-edgeA[0];
  LHS.col(0)= (edgeA[1]-edgeA[0]);
  LHS.col(1)=-(edgeB[1]-edgeB[0]);
  if(fabs(LHS.determinant())<Epsilon<T>::defaultEps()) {
    return false;   //parallel line segment, doesn't matter
  } else {
    Vec2T st=LHS.inverse()*RHS;
    return st[0]>=0 && st[0]<=1 && st[1]>=0 && st[1]<=1;
  }
}
//helper
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

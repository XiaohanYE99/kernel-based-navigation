#include "SpatialHashLinkedList.h"
#include "BBox.h"
#include <stack>

#ifdef WIN32
#pragma warning(push)
#pragma warning(disable:4456)
#pragma warning(disable:4458)
#endif
namespace RVO {
SpatialHashLinkedList::SpatialHashLinkedList():_locked(false) {}
SpatialHashLinkedList::~SpatialHashLinkedList() {
  for(int i=0; i<(int)_writelock.size(); i++)
    omp_destroy_lock(&_writelock[i]);
}
void SpatialHashLinkedList::lock() {
  _locked=true;
}
void SpatialHashLinkedList::unlock() {
  _locked=false;
}
void SpatialHashLinkedList::buildSpatialHash(VecCM pos0,VecCM pos1,T R,bool useHash) {
  if(_locked)
    return;
  //compute bounds
  _nodes.resize(_vss.size());
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_vss.size(); i++) {
    _nodes[i]._bb=BBox();
    if(pos0.data())
      _nodes[i]._bb.setUnion(_vss[i]->operator()(pos0));
    if(pos1.data())
      _nodes[i]._bb.setUnion(_vss[i]->operator()(pos1));
    _nodes[i]._radius=_nodes[i]._bb.getExtent().norm()/2+R;
    _nodes[i]._ctr=(_nodes[i]._bb._minC+_nodes[i]._bb._maxC)/2;
    _nodes[i]._next=-1;
  }
  if(!useHash)
    return;
  //find range
  _nodesBkg=_nodes;
  reduce([&](SpatialHashNode& a,SpatialHashNode& b) {
    a._radius=std::max(a._radius,b._radius);
    a._bb=a._bb.getUnion(b._bb);
  });
  _R=_nodes[0]._radius;
  _bb=_nodes[0]._bb;
  _nodes.swap(_nodesBkg);
  //compute parameter
  _invR=1/_R;
  _nrCell.array()=(_bb.getExtent()*_invR).cwiseMax(Vec2T::Ones()).array().ceil().cast<int>();
  _stride[0]=_nrCell[1];
  _stride[1]=1;
  //relink
  _heads.assign(_nrCell.prod(),-1);
  int nrLock=(int)_writelock.size();
  if(_nrCell.prod()>nrLock) {
    _writelock.resize(_nrCell.prod(),omp_lock_t());
    for(int i=nrLock; i<(int)_writelock.size(); i++)
      omp_init_lock(&_writelock[i]);
  }
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_vss.size(); i++) {
    int index=hashOff(_nodes[i]._ctr);
    omp_set_lock(&_writelock[index]);
    _nodes[i]._next=_heads[index];
    _heads[index]=i;
    omp_unset_lock(&_writelock[index]);
  }
}
void SpatialHashLinkedList::detectImplicitShape(std::function<bool(AgentObstacleNeighbor)> VVss,const BoundingVolumeHierarchy& bvh,T margin) {
  if(bvh.getNodes().empty())
    return;
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_nodes.size(); i++) {
    BBox bb(_nodes[i]._ctr);
    bb.enlarged(_nodes[i]._radius+margin);
    //stack
    std::stack<int> ss;
    ss.push((int)bvh.getNodes().size()-1);
    while(!ss.empty()) {
      int curr=ss.top();
      ss.pop();
      if(!bvh.getNodes()[curr]._bb.intersect(bb))
        continue;
      else if(bvh.getNodes()[curr]._cell>=0) {
        AgentObstacleNeighbor VV;
        VV._v=_vss[i];
        VV._o=bvh.getObstacles()[bvh.getNodes()[curr]._cell];
        VVss(VV);
      } else {
        ss.push(bvh.getNodes()[curr]._l);
        ss.push(bvh.getNodes()[curr]._r);
      }
    }
  }
}
void SpatialHashLinkedList::detectImplicitShapeBF(std::function<bool(AgentObstacleNeighbor)> VVss,const BoundingVolumeHierarchy& bvh,T margin) {
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_nodes.size(); i++) {
    BBox bb(_nodes[i]._ctr);
    bb.enlarged(_nodes[i]._radius+margin);
    //loop over all implicit shapes
    std::stack<int> ss;
    ss.push((int)bvh.getNodes().size()-1);
    for(int j=0; j<(int)bvh.getNodes().size(); j++)
      if(bvh.getNodes()[j]._cell>=0) {
        if(!bvh.getNodes()[j]._bb.intersect(bb))
          continue;
        AgentObstacleNeighbor VV;
        VV._v=_vss[i];
        VV._o=bvh.getObstacles()[bvh.getNodes()[j]._cell];
        VVss(VV);
      }
  }
}
void SpatialHashLinkedList::detectSphereBroad(std::function<bool(AgentNeighbor)> VVss,const SpatialHash& otherSH,T margin) {
  const SpatialHashLinkedList& other=dynamic_cast<const SpatialHashLinkedList&>(otherSH);
  bool selfCollision=this==&other;
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)other._vss.size(); i++) {
    AgentNeighbor VV;
    T searchRange=other._nodes[i]._radius+margin+_R;
    Eigen::Matrix<int,2,1> L=hash(other._nodes[i]._ctr-Vec2T::Constant(searchRange)).cwiseMax(Eigen::Matrix<int,2,1>::Zero());
    Eigen::Matrix<int,2,1> U=hash(other._nodes[i]._ctr+Vec2T::Constant(searchRange)).cwiseMin(_nrCell-Eigen::Matrix<int,2,1>::Ones());
    for(int x=L[0],offX=L.dot(_stride); x<=U[0]; x++,offX+=_stride[0])
      for(int y=L[1],offY=offX; y<=U[1]; y++,offY+=_stride[1]) {
        int head=_heads[offY];
        while(head>=0) {
          VV._v[0]=other._vss[i];
          VV._v[1]=_vss[head];
          if(!selfCollision || VV._v[0]<VV._v[1])
            if((other._nodes[i]._ctr-_nodes[head]._ctr).norm()<other._nodes[i]._radius+_nodes[head]._radius+margin)
              VVss(VV);
          head=_nodes[head]._next;
        }
      }
  }
}
void SpatialHashLinkedList::detectSphereBroadBF(std::function<bool(AgentNeighbor)> VVss,const SpatialHash& otherSH,T margin) {
  const SpatialHashLinkedList& other=dynamic_cast<const SpatialHashLinkedList&>(otherSH);
  bool selfCollision=this==&other;
  _VVCheckList.clear();
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)other._vss.size(); i++)
    for(int j=0; j<(int)_vss.size(); j++) {
      AgentNeighbor VV;
      VV._v[0]=other._vss[i];
      VV._v[1]=_vss[j];
      if(!selfCollision || VV._v[0]<VV._v[1])
        if((other._nodes[i]._ctr-_nodes[j]._ctr).norm()<other._nodes[i]._radius+_nodes[j]._radius+margin)
          OMP_CRITICAL_
          _VVCheckList.push_back(VV);
    }
  //make unique
  _VVUniqueCheckList.assign(_VVCheckList.begin(),_VVCheckList.end());
  std::sort(_VVUniqueCheckList.begin(),_VVUniqueCheckList.end());
  _VVUniqueCheckList.erase(std::unique(_VVUniqueCheckList.begin(),_VVUniqueCheckList.end()),_VVUniqueCheckList.end());
  for(const auto& VV:_VVUniqueCheckList)
    VVss(VV);
}
//helper
int SpatialHashLinkedList::hashOff(const Vec2T& pt) const {
  return hash(pt).dot(_stride);
}
Eigen::Matrix<int,2,1> SpatialHashLinkedList::hash(const Vec2T& pt) const {
  return ((pt-_bb._minC)*_invR).array().floor().matrix().cast<int>().cwiseMin(_nrCell-Eigen::Matrix<int,2,1>::Ones());
}
void SpatialHashLinkedList::reduce(std::function<void(SpatialHashNode&,SpatialHashNode&)> op) {
  for(int off=2,offHalf=1; offHalf<(int)_vss.size(); off<<=1,offHalf<<=1) {
    OMP_PARALLEL_FOR_
    for(int i=0; i<(int)_vss.size(); i+=off)
      if(i+offHalf<(int)_vss.size())
        op(_nodes[i],_nodes[i+offHalf]);
  }
}
}
#ifdef WIN32
#pragma warning(pop)
#endif

#include "SpatialHashRadixSort.h"
#include "BBox.h"
#include <stack>

#ifdef WIN32
#pragma warning(push)
#pragma warning(disable:4456)
#pragma warning(disable:4458)
#endif
namespace RVO {
SpatialHashRadixSort::SpatialHashRadixSort():_locked(false) {}
void SpatialHashRadixSort::lock() {
  _locked=true;
}
void SpatialHashRadixSort::unlock() {
  _locked=false;
}
void SpatialHashRadixSort::buildSpatialHash(VecCM pos0,VecCM pos1,T R) {
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
  //sort
  _indices.resize(_vss.size());
  _offsetsInv.resize(_vss.size());
  _offsets.resize(_vss.size());
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_vss.size(); i++) {
    _indices[i]=hashOff(_nodes[i]._ctr);
    _offsetsInv[i]=i;
  }
  radixSort(_indices.data(),_offsetsInv.data(),(int)_vss.size());
  //mark start/end
  _starts.assign(_nrCell.prod(),-1);
  _ends.assign(_nrCell.prod(),-1);
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_vss.size(); i++) {
    _offsets[_offsetsInv[i]]=i;
    if(i==0)
      _starts[_indices[i]]=i;
    else if(_indices[i]!=_indices[i-1])
      _ends[_indices[i-1]]=_starts[_indices[i]]=i;
    if(i==(int)_vss.size()-1)
      _ends[_indices[i]]=i+1;
  }
//#define DEBUG_SPATIAL_HASH
#ifdef DEBUG_SPATIAL_HASH
  int total=0;
  for(int i=0; i<(int)_starts.size(); i++) {
    total+=_ends[i]-_starts[i];
    for(int start=_starts[i],end=_ends[i]; start<end; start++) {
      ASSERT_MSG(_indices[start]==i,"SpatialHash hash mismatch!")
      ASSERT_MSG(hashOff(_nodes[_offsets[start]]._ctr)==i,"SpatialHash hashOff mismatch!")
    }
  }
  ASSERT_MSG(total==(int)_vss.size(),"SpatialHash size mismatch!")
#endif
}
void SpatialHashRadixSort::detectImplicitShape(std::function<bool(AgentObstacleNeighbor)> VVss,const BoundingVolumeHierarchy& bvh,T margin) {
  if(bvh.getNodes().empty())
    return;
  //OMP_PARALLEL_FOR_
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
void SpatialHashRadixSort::detectImplicitShapeBF(std::function<bool(AgentObstacleNeighbor)> VVss,const BoundingVolumeHierarchy& bvh,T margin) {
  //OMP_PARALLEL_FOR_
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
void SpatialHashRadixSort::detectSphereBroad(std::function<bool(AgentNeighbor)> VVss,const SpatialHash& otherSH,T margin) {
  const SpatialHashRadixSort& other=dynamic_cast<const SpatialHashRadixSort&>(otherSH);
  bool selfCollision=this==&other;
  //OMP_PARALLEL_FOR_
  for(int i=0; i<(int)other._vss.size(); i++) {
    AgentNeighbor VV;
    T searchRange=other._nodes[i]._radius+margin+_R;
    Eigen::Matrix<int,2,1> L=hash(other._nodes[i]._ctr-Vec2T::Constant(searchRange)).cwiseMax(Eigen::Matrix<int,2,1>::Zero());
    Eigen::Matrix<int,2,1> U=hash(other._nodes[i]._ctr+Vec2T::Constant(searchRange)).cwiseMin(_nrCell-Eigen::Matrix<int,2,1>::Ones());
    for(int x=L[0],offX=L.dot(_stride); x<=U[0]; x++,offX+=_stride[0])
      for(int y=L[1],offY=offX; y<=U[1]; y++,offY+=_stride[1])
        for(int start=_starts[offY],end=_ends[offY]; start<end; start++) {
          VV._v[0]=other._vss[i];
          VV._v[1]=_vss[_offsets[start]];
          if(!selfCollision || VV._v[0]<VV._v[1])
            if((other._nodes[i]._ctr-_nodes[_offsets[start]]._ctr).norm()<other._nodes[i]._radius+_nodes[_offsets[start]]._radius+margin)
              VVss(VV);
        }
  }
}
void SpatialHashRadixSort::detectSphereBroadBF(std::function<bool(AgentNeighbor)> VVss,const SpatialHash& otherSH,T margin) {
  const SpatialHashRadixSort& other=dynamic_cast<const SpatialHashRadixSort&>(otherSH);
  bool selfCollision=this==&other;
  _VVCheckList.clear();
  //OMP_PARALLEL_FOR_
  for(int i=0; i<(int)other._vss.size(); i++)
    for(int j=0; j<(int)_vss.size(); j++) {
      AgentNeighbor VV;
      VV._v[0]=other._vss[i];
      VV._v[1]=_vss[j];
      if(!selfCollision || VV._v[0]<VV._v[1])
        if((other._nodes[i]._ctr-_nodes[j]._ctr).norm()<other._nodes[i]._radius+_nodes[j]._radius+margin)
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
int SpatialHashRadixSort::hashOff(const Vec2T& pt) const {
  return hash(pt).dot(_stride);
}
Eigen::Matrix<int,2,1> SpatialHashRadixSort::hash(const Vec2T& pt) const {
  return ((pt-_bb._minC)*_invR).array().floor().matrix().cast<int>().cwiseMin(_nrCell-Eigen::Matrix<int,2,1>::Ones());
}
void SpatialHashRadixSort::reduce(std::function<void(SpatialHashNode&,SpatialHashNode&)> op) {
  for(int off=2,offHalf=1; offHalf<(int)_vss.size(); off<<=1,offHalf<<=1) {
    OMP_PARALLEL_FOR_
    for(int i=0; i<(int)_vss.size(); i+=off)
      if(i+offHalf<(int)_vss.size())
        op(_nodes[i],_nodes[i+offHalf]);
  }
}
}
#include "pradsort.hpp"
void radixSort(int* val,int* key,int N) {
  prsort::pradsort(val,key,N,8,NULL);
}
#ifdef WIN32
#pragma warning(pop)
#endif

#ifndef SPATIAL_HASH_LINKED_LIST_H
#define SPATIAL_HASH_LINKED_LIST_H

#include "SpatialHash.h"

namespace RVO {
class SpatialHashLinkedList : public SpatialHash {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  DECL_MAP_FUNCS
  struct SpatialHashNode {
    Vec2T _ctr;
    BBox _bb;
    T _radius;
    int _next;
  };
  SpatialHashLinkedList();
  ~SpatialHashLinkedList();
  void lock() override;
  void unlock() override;
  void buildSpatialHash(VecCM pos0,VecCM pos1,T R,bool useHash=true) override;
  void detectImplicitShape(std::function<bool(AgentObstacleNeighbor)> VVss,const BoundingVolumeHierarchy& bvh,T margin) override;
  void detectImplicitShapeBF(std::function<bool(AgentObstacleNeighbor)> VVss,const BoundingVolumeHierarchy& bvh,T margin) override;
  void detectSphereBroad(std::function<bool(AgentNeighbor)> VVss,const SpatialHash& other,T margin) override;
  void detectSphereBroadBF(std::function<bool(AgentNeighbor)> VVss,const SpatialHash& other,T margin) override;
 protected:
  int hashOff(const Vec2T& pt) const;
  Eigen::Matrix<int,2,1> hash(const Vec2T& pt) const;
  void reduce(std::function<void(SpatialHashNode&,SpatialHashNode&)> op);
  //temporary data, not serialized
  T _R,_invR;
  BBox _bb;
  bool _locked;
  std::vector<omp_lock_t> _writelock;
  Eigen::Matrix<int,2,1> _stride,_nrCell;
  std::vector<SpatialHashNode> _nodes,_nodesBkg;
  std::vector<int> _heads;
  //temporary data, not serialized
  std::vector<AgentNeighbor> _VVUniqueCheckList;
  std::vector<AgentNeighbor> _VVCheckList;
};
}

#endif

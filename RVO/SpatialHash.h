#ifndef SPATIAL_HASH_H
#define SPATIAL_HASH_H

#include "Agent.h"
#include "BBox.h"
#include <vector>
#include "BoundingVolumeHierarchy.h"

namespace RVO {
class SpatialHash {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  DECL_MAP_FUNCS
  virtual void lock()=0;
  virtual void unlock()=0;
  virtual void buildSpatialHash(VecCM pos0,VecCM pos1,T R,bool useHash=true)=0;
  virtual void detectImplicitShape(std::function<bool(AgentObstacleNeighbor)> VVss,const BoundingVolumeHierarchy& bvh,T margin)=0;
  virtual void detectImplicitShapeBF(std::function<bool(AgentObstacleNeighbor)> VVss,const BoundingVolumeHierarchy& bvh,T margin)=0;
  virtual void detectSphereBroad(std::function<bool(AgentNeighbor)> VVss,const SpatialHash& other,T margin)=0;
  virtual void detectSphereBroadBF(std::function<bool(AgentNeighbor)> VVss,const SpatialHash& other,T margin)=0;
  const std::vector<std::shared_ptr<Agent>>& vss() const;
  void addVertex(std::shared_ptr<Agent> v);
 protected:
  std::vector<std::shared_ptr<Agent>> _vss;
};
}

#endif

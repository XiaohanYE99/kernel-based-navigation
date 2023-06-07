#ifndef SOURCE_SINK_H
#define SOURCE_SINK_H

#include "RVO.h"

namespace RVO {
struct Trajectory {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  Trajectory();
  bool _terminated;
  std::vector<Vec2T> _pos;
  Vec2T _target;
  T _rad;
};
class SourceSink {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_I
  DECL_MAT_VEC_MAP_TYPES_T
  DECL_MAP_FUNCS
  SourceSink(T maxVelocity,int maxBatch);
  std::vector<Trajectory> getTrajectories() const;
  void addSourceSink(const Vec2T& source,const Vec2T& target,const BBox& sink,T rad);
  void addAgents(RVOSimulator& sim,T eps=1e-4);
  void recordAgents(const RVOSimulator& sim);
  void removeAgents(RVOSimulator& sim);
  void reset();
 private:
  DynamicMat<T> _sourcePos;
  DynamicMat<T> _targetPos;
  DynamicMat<T> _sinkRegion;
  DynamicVec<T> _rad;
  DynamicVec<int> _id;
  T _maxVelocity;
  int _maxBatch;
  //recorded trajectories
  std::vector<Trajectory> _trajectories;
};
}

#endif

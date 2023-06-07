#ifndef SOURCE_SINK_H
#define SOURCE_SINK_H

#include "RVO.h"

namespace RVO {
struct State {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  Vec2T _pos,_vel,_rad;
};
class SourceSink {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_I
  DECL_MAT_VEC_MAP_TYPES_T
  DECL_MAP_FUNCS
  void addSourceSink(const Vec2T& source,const Vec2T& target,const BBox& sink,T rad);
  void removeAgents(RVOSimulator& sim);
  void addAgents(RVOSimulator& sim,T eps=1e-4);
  void reset(T maxVelocity);
 private:
  DynamicMat<T> _sourcePos;
  DynamicMat<T> _targetPos;
  DynamicMat<T> _sinkRegion;
  DynamicVec<T> _rad;
  DynamicVec<int> _id;
  T _maxVelocity;
  //recorded trajectories
  std::vector<std::vector<State>> _trajectories;
};
}

#endif

#ifndef SOURCE_SINK_H
#define SOURCE_SINK_H

#include "RVO.h"

namespace RVO {
struct Trajectory {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  friend class SourceSink;
  Trajectory();
  int startFrame() const;
  int endFrame() const;
  bool terminated() const;
#ifdef SWIG
  std::vector<Eigen::Matrix<double,2,1>> pos() const;
  Eigen::Matrix<double,2,1> target() const;
  double rad() const;
#else
  std::vector<Vec2T> pos() const;
  Vec2T target() const;
  T rad() const;
#endif
 private:
  int _startFrame;
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
#ifndef SWIG
  void addSourceSink(const Vec2T& source,const Vec2T& target,const BBox& sink,T rad);
  static std::pair<Mat2XT,Vec> getAgentPositions(int frameId,const std::vector<Trajectory>& trajectories);
#endif
  void addAgents(int frameId,RVOSimulator& sim,T eps=1e-4);
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

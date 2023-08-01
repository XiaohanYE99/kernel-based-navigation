#ifndef SOURCE_SINK_H
#define SOURCE_SINK_H

#include "RVO.h"

namespace RVO {
struct Trajectory {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  Trajectory();
  Trajectory(bool recordFull,int frameId,const Vec2T& target,T r,int id);
  int startFrame() const;
  int endFrame() const;
  bool isFullTrajectory() const;
  bool terminated() const;
  void terminate();
  void addPos(const Vec2T& pos);
  Vec2T pos(int frameId) const;
  Mat2XT pos() const;
  Vec2T target() const;
  T rad() const;
  short sid() const;
  short aid() const;
 private:
  int _endFrame;
  int _startFrame;
  bool _terminated,_recordFull;
  std::vector<Vec2T> _fullPos;
  Mat2T _startEndPos;
  Vec2T _target;
  T _rad;
  int _id;
};
class SourceSink {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_I
  DECL_MAT_VEC_MAP_TYPES_T
  DECL_MAP_FUNCS
  typedef std::tuple<Mat2XT,Vec,Veci> Frame;
  SourceSink(T maxVelocity,int maxBatch,bool recordFull);
  const DynamicMat<T>& getSourcePos() const;
  const DynamicMat<T>& getTargetPos() const;
  const DynamicMat<T>& getSinkRegion() const;
  std::vector<Trajectory> getTrajectories() const;
  void addSourceSink(const Vec2T& source,const Vec2T& target,const BBox& sink,T rad);
  static Frame getAgentPositions(int frameId,const std::vector<Trajectory>& trajectories);
  static int assembleAgentId(short sid,short aid);
  static short extractSourceId(int id);
  static short extractAgentId(int id);
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
  bool _recordFull;
  //recorded trajectories
  std::vector<Trajectory> _trajectories;
};
}

#endif

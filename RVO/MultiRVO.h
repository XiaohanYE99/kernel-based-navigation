#ifndef MULTI_RVO_H
#define MULTI_RVO_H

#include "RVO.h"
#include "SourceSink.h"

namespace RVO {
class MultiRVOSimulator {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
#ifndef SWIG
  DECL_MAP_FUNCS
#endif
  MultiRVOSimulator(int batchSize,T d0=1,T gTol=1e-4,T coef=1,T timestep=1,int maxIter=1000,bool radixSort=false,bool useHash=true,const std::string& optimizer="NEWTON");
  void clearAgent();
  void clearObstacle();
  int getNrObstacle() const;
#ifdef SWIG
  std::vector<Eigen::Matrix<double,2,1>> getObstacle(int i) const;
#else
  std::vector<Vec2T> getObstacle(int i) const;
#endif
  int getNrAgent() const;
  void resetSourceSink(T maxVelocity,int maxBatch);
  void addSourceSink(Vec2T source,Vec2T target,Vec2T minC,Vec2T maxC,T rad);
#ifdef SWIG
  std::vector<Eigen::Matrix<double,2,1>> getAgentPosition(int i) const;
  std::vector<Eigen::Matrix<double,2,1>> getAgentVelocity(int i) const;
  std::vector<Eigen::Matrix<double,2,2>> getAgentDVDP(int i) const;
  int addAgent(std::vector<Eigen::Matrix<double,2,1>> pos,std::vector<Eigen::Matrix<double,2,1>> vel,std::vector<double> rad);
  void setAgentPosition(int i,std::vector<Eigen::Matrix<double,2,1>> pos);
  void setAgentVelocity(int i,std::vector<Eigen::Matrix<double,2,1>> vel);
  void setAgentTarget(int i,std::vector<Eigen::Matrix<double,2,1>> target,T maxVelocity);
  int addObstacle(std::vector<Eigen::Matrix<double,2,1>> vss);
#else
  std::vector<Vec2T> getAgentPosition(int i) const;
  std::vector<Vec2T> getAgentVelocity(int i) const;
  std::vector<Mat2T> getAgentDVDP(int i) const;
  std::vector<T> getAgentRadius(int i) const;
  int addAgent(std::vector<Vec2T> pos,std::vector<Vec2T> vel,std::vector<T> rad);
  void setAgentPosition(int i,std::vector<Vec2T> pos);
  void setAgentVelocity(int i,std::vector<Vec2T> vel);
  void setAgentTarget(int i,std::vector<Vec2T> target,T maxVelocity);
  int addObstacle(std::vector<Vec2T> vss);
#endif
  void buildVisibility();
  void clearVisibility();
  void setNewtonParameter(int maxIter,T gTol,T d0,T coef=1);
  void setTimestep(T timestep);
  T timestep() const;
  int getBatchSize() const;
  RVOSimulator& getSubSimulator(int id);
  const RVOSimulator& getSubSimulator(int id) const;
  std::vector<char> optimize(bool requireGrad,bool output);
  void updateAgentTargets();
#ifdef SWIG
  std::vector<Eigen::Matrix<double,-1,-1>> getDXDX() const;
  std::vector<Eigen::Matrix<double,-1,-1>> getDXDV() const;
#else
  std::vector<MatT> getDXDX() const;
  std::vector<MatT> getDXDV() const;
#endif
 private:
  std::vector<RVOSimulator> _sims;
  std::vector<SourceSink> _sss;
};
}

#endif

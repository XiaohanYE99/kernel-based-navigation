#ifndef MULTI_RVO_H
#define MULTI_RVO_H

#include "RVO.h"

namespace RVO {
class MultiRVOSimulator {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  DECL_MAP_FUNCS
  MultiRVOSimulator(int batchSize,T rad,T d0=1,T gTol=1e-4,T coef=1,T timestep=1,int maxIter=1000,bool radixSort=false,bool useHash=true);
  T getRadius() const;
  void clearAgent();
  void clearObstacle();
  int getNrObstacle() const;
  std::vector<Vec2T> getObstacle(int i) const;
  int getNrAgent() const;
  std::vector<Vec2T> getAgentPosition(int i) const;
  std::vector<Vec2T> getAgentVelocity(int i) const;
  int addAgent(const std::vector<Vec2T>& pos,const std::vector<Vec2T>& vel);
  void setAgentPosition(int i,const std::vector<Vec2T>& pos);
  void setAgentVelocity(int i,const std::vector<Vec2T>& vel);
  void setAgentTarget(int i,const std::vector<Vec2T>& target,T maxVelocity);
  void addObstacle(const std::vector<Vec2T>& vss);
  void setNewtonParameter(int maxIter,T gTol,T d0,T coef=1);
  void setAgentRadius(T radius);
  void setTimestep(T timestep);
  T timestep() const;
  std::vector<char> optimize(std::vector<MatT>* DXDV,std::vector<MatT>* DXDX,bool output);
 private:
  std::vector<RVOSimulator> _sims;
};
}

#endif

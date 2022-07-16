#ifndef RVO_H
#define RVO_H

#include "SpatialHash.h"
#include "BoundingVolumeHierarchy.h"
#include "ParallelVector.h"
#include <Eigen/Sparse>

namespace RVO {
class RVO {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  DECL_MAP_FUNCS
  typedef Eigen::Triplet<T,int> STrip;
  typedef ParallelVector<STrip> STrips;
  typedef Eigen::SparseMatrix<T,0,int> SMatT;
  RVO(T rad,T d0=1,T gTol=1e-4,T coef=1,int maxIter=100);
  void clearAgent();
  void clearObstacle();
  const Mat2XT& getAgent() const;
  int addAgent(const Vec2T& pos,const Vec2T& vel);
  void setAgent(int i,const Vec2T& pos,const Vec2T& vel);
  void addObstacle(const std::vector<Vec2T>& vss);
  void setNewtonParameter(int maxIter,T gTol,T d0,T coef=1);
  void setAgentRadius(T radius);
  void setTimestep(T timestep);
  T timestep() const;
  bool optimize(MatT* DXDV,MatT* DXDX,bool output);
 private:
  static T clog(T d,T* D,T* DD,T d0,T coef);
  bool lineSearch(T E,const Vec& g,const Vec& d,T& alpha,Vec& newX,
                  std::function<bool(const Vec&,T&)> eval,T alphaMin) const;
  bool energy(const Vec& prevPos,const Vec& pos,T* f,Vec* g,SMatT* h);
  bool energyAA(int aid,int bid,const Vec2T& a,const Vec2T& b,T* f,Vec* g,STrips* trips) const;
  bool energyAO(int aid,const Vec2T& a,const Vec2T o[2],T* f,Vec* g,STrips* trips) const;
  bool intersect(const Vec2T edgeA[2],const Vec2T edgeB[2]) const;
  template <typename MAT>
  static void addBlock(STrips& trips,int r,int c,const MAT& blk);
  static T absMax(const SMatT& h);
  std::shared_ptr<SpatialHash> _hash;
  BoundingVolumeHierarchy _bvh;
  Mat2XT _perfVelocities;
  Mat2XT _agentPositions;
  T _timestep,_gTol,_d0,_coef,_rad;
  int _maxIter;
  SMatT _id;
};
}

#endif

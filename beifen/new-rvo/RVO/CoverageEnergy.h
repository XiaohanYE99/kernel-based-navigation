#ifndef COVERAGE_ENERGY_H
#define COVERAGE_ENERGY_H

#include "RVO.h"

namespace RVO {
class CoverageEnergy {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
#ifndef SWIG
  DECL_MAP_FUNCS
  typedef Eigen::Triplet<T,int> STrip;
  typedef ParallelVector<STrip> STrips;
  typedef Eigen::SparseMatrix<T,0,int> SMatT;
#endif
  CoverageEnergy(const RVOSimulator& sim,T range,bool visibleOnly=true);
#ifndef SWIG
  T loss(Vec pos);
  Vec grad() const;
#else
  double loss(Eigen::Matrix<double,-1,1> pos);
  Eigen::Matrix<double,-1,1> grad() const;
#endif
  void debugCoverage(T scale);
#ifndef SWIG
  void energy(VecCM pos,T* f,Vec* g,STrips* h,Eigen::Matrix<int,4,1>& nBarrier);
  void energy(VecCM pos,T* f,Vec* g,SMatT* h,Eigen::Matrix<int,4,1>& nBarrier);
 private:
  void energyAA(int aid,int bid,const Vec2T& a,const Vec2T& b,T* f,Vec* g,STrips* trips,Eigen::Matrix<int,4,1>& nBarrier) const;
  void energyAO(int aid,const Vec2T& a,const Vec2T o[2],T* f,Vec* g,STrips* trips,Eigen::Matrix<int,4,1>& nBarrier) const;
  static T kernel(T distSq,T* D,T* DD,T range);
  std::shared_ptr<SpatialHash> _hash;
  const BoundingVolumeHierarchy& _bvh;
  bool _visibleOnly,_useHash;
  T _range;
  Vec _grad;
#endif
};
}

#endif

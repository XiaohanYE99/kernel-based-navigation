#ifndef ORCA_H
#define ORCA_H

#include "RVO.h"
#include <unsupported/Eigen/AutoDiff>

namespace RVO {
#ifndef SWIG
struct VelocityObstacle {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  typedef Eigen::Matrix<T,10,1> Derivative;
  typedef Eigen::AutoDiffScalar<Derivative> AD;
  typedef Eigen::Matrix<AD,2,2> Mat2TAD;
  typedef Eigen::Matrix<AD,2,1> Vec2TAD;
  Vec2T proj(const Vec2T& v,T tol) const;
  Vec2T proj(const Vec2T& v) const;
  T violation(const Vec2T& v) const;
  bool outside(const Vec2T& v) const;
  Vec2T pos() const;
  Vec2T nor() const;
  Mat2T DposDpa() const;
  Mat2T DposDpb() const;
  Mat2T DposDva() const;
  Mat2T DposDvb() const;
  Mat2T DnorDpa() const;
  Mat2T DnorDpb() const;
  Mat2T DnorDva() const;
  Mat2T DnorDvb() const;
  Vec2TAD _pos,_nor;    //_vab is a point on the boundary, _nab is the outward-pointing normal
  int _aid,_bid;
  int _case;
};
struct LPSolution {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  std::pair<int,int> _activeSet;
  Vec2T _vIn,_vOut;
  bool _succ;
};
#endif
class ORCASimulator : public RVOSimulator {
 public:
#ifndef SWIG
  typedef Eigen::Matrix<T,10,1> Derivative;
  typedef Eigen::AutoDiffScalar<Derivative> AD;
  typedef Eigen::Matrix<AD,2,2> Mat2TAD;
  typedef Eigen::Matrix<AD,2,1> Vec2TAD;
#endif
  ORCASimulator(const ORCASimulator& other);
#ifndef SWIG
  ORCASimulator& operator=(const ORCASimulator& other);
#endif
  ORCASimulator(T rad,T d0=1,T gTol=1e-4,T coef=1,T timestep=1,int maxIter=1000,bool radixSort=false,bool useHash=true);
  virtual bool optimize(bool requireGrad,bool output) override;
#ifndef SWIG
  void debugVO(int aid,int bid,int testCase,T eps=1e-4f);
  void debugVO(int aid,int testCase,bool mustInside,T eps=1e-4f);
  void debugDerivatives(const VelocityObstacle& VO);
  void debugDerivatives(const VelocityObstacle& VO,const Vec2T o[2]);
 protected:
  VelocityObstacle computeVelocityObstacle(int aid,int bid) const;
  VelocityObstacle computeVelocityObstacle(int aid,const Vec2T o[2]) const;
  void computeVelocityObstacle(VelocityObstacle& ret,const Vec2TAD& o,const Vec2TAD& a,const Vec2TAD& dL,const Vec2TAD& dR) const;
  void computeVelocityObstacle(VelocityObstacle& ret,const Vec2TAD o[2],const Vec2TAD a[2],const Vec2TAD& dL,const Vec2TAD& dR) const;
  static void buildGrad(int id,MatT& DVDX,MatT& DVDV,const LPSolution& sol,const std::vector<VelocityObstacle>& VO,T tol);
  static bool solveActiveSet(std::pair<int,int>& activeSetInOut,Vec2T& vInOut,const std::vector<VelocityObstacle>& VO,int i,int j,T tol);
  static bool updateActiveSet(LPSolution& sol,const std::vector<VelocityObstacle>& VO,int i,T tol);
  static LPSolution solveLP(const Vec2T& v,const std::vector<VelocityObstacle>& VO,T tol);
  static T violation(const Vec2T& vOut,const std::vector<VelocityObstacle>& VO);
  static T cross2D(const Vec2TAD& a,const Vec2TAD& b);
  static T cross2D(const Vec2T& a,const Vec2T& b);
  static Mat2TAD rot2D(AD dist,AD cord);
  //data
  std::vector<std::vector<VelocityObstacle>> _LPs;
  std::vector<LPSolution> _LPSolutions;
#endif
};
}

#endif

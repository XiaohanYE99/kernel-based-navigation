#ifndef RVO_H
#define RVO_H

#include "BoundingVolumeHierarchy.h"
#include "ParallelVector.h"
#include <unordered_map>
#include <Eigen/Sparse>

namespace RVO {
class SpatialHash;
class RVOSimulator {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
#ifndef SWIG
  DECL_MAP_FUNCS
  typedef Eigen::Triplet<T,int> STrip;
  typedef ParallelVector<STrip> STrips;
  typedef Eigen::SparseMatrix<T,0,int> SMatT;
#endif
  RVOSimulator(const RVOSimulator& other);
  RVOSimulator& operator=(const RVOSimulator& other);
  RVOSimulator(T rad,T d0=1,T gTol=1e-4,T coef=1,T timestep=1,int maxIter=1000,bool radixSort=false,bool useHash=true);
  bool getUseHash() const;
  T getRadius() const;
  void clearAgent();
  void clearObstacle();
  int getNrObstacle() const;
#ifdef SWIG
  std::vector<Eigen::Matrix<double,2,1>> getObstacle(int i) const;
#else
  std::vector<Vec2T> getObstacle(int i) const;
#endif
  int getNrAgent() const;
#ifndef SWIG
  Mat2XT& getAgentPositions();
  Mat2XT& getAgentVelocities();
#endif
  Mat2XT getAgentPositions() const;
  Mat2XT getAgentVelocities() const;
  Vec2T getAgentPosition(int i) const;
  Vec2T getAgentVelocity(int i) const;
#ifdef SWIG
  int addAgent(const Eigen::Matrix<double,2,1>& pos,const Eigen::Matrix<double,2,1>& vel);
  void setAgentPosition(int i,const Eigen::Matrix<double,2,1>& pos);
  void setAgentVelocity(int i,const Eigen::Matrix<double,2,1>& vel);
  void setAgentTarget(int i,const Eigen::Matrix<double,2,1>& target,T maxVelocity);
  int addObstacle(std::vector<Eigen::Matrix<double,2,1>> vss);
#else
  int addAgent(const Vec2T& pos,const Vec2T& vel);
  void setAgentPosition(int i,const Vec2T& pos);
  void setAgentVelocity(int i,const Vec2T& vel);
  void setAgentTarget(int i,const Vec2T& target,T maxVelocity);
  int addObstacle(std::vector<Vec2T> vss);
#endif
  void setNewtonParameter(int maxIter,T gTol,T d0,T coef=1);
  void setAgentRadius(T radius);
  void setTimestep(T timestep);
  T timestep() const;
  bool optimize(bool requireGrad,bool output);
  void updateAgentTargets();
  MatT getDXDX() const;
  MatT getDXDV() const;
  void debugNeighbor(T scale);
  void debugEnergy(T scale,T dscale=1);
#ifndef SWIG
  std::shared_ptr<SpatialHash> getHash() const;
  const BoundingVolumeHierarchy& getBVH() const;
  static void addBlock(Vec& g,int r,const Vec2T& blk);
  template <typename MAT>
  static void addBlock(STrips& trips,int r,int c,const MAT& blk) {
    for(int R=0; R<blk.rows(); R++)
      for(int C=0; C<blk.cols(); C++)
        trips.push_back(STrip(r+R,c+C,blk(R,C)));
  }
  static T absMax(const SMatT& h);
 private:
  static T clog(T d,T* D,T* DD,T d0,T coef);
  bool lineSearch(T E,const Vec& g,const Vec& d,T& alpha,Vec& newX,
                  std::function<bool(const Vec&,T&)> eval,T alphaMin) const;
  bool energy(VecCM prevPos,VecCM pos,T* f,Vec* g,SMatT* h,Eigen::Matrix<int,4,1>& nBarrier);
  bool energyAA(int aid,int bid,const Vec2T& a,const Vec2T& b,T* f,Vec* g,STrips* trips,Eigen::Matrix<int,4,1>& nBarrier) const;
  bool energyAO(int aid,const Vec2T& a,const Vec2T o[2],T* f,Vec* g,STrips* trips,Eigen::Matrix<int,4,1>& nBarrier) const;
  std::shared_ptr<SpatialHash> _hash;
  BoundingVolumeHierarchy _bvh;
  Mat2XT _perfVelocities;
  Mat2XT _agentPositions;
  Eigen::SimplicialLDLT<SMatT> _sol;
  std::unordered_map<int,Vec3T> _agentTargets;
  T _timestep,_gTol,_d0,_coef,_rad;
  bool _useHash;
  int _maxIter;
  //data
  MatT _DXDX,_DXDV;
  SMatT _id;
#endif
};
}

#endif

#ifndef RVO_H
#define RVO_H

#include "BoundingVolumeHierarchy.h"
#include "ParallelVector.h"
#include "LBFGSUpdate.h"
#include <unordered_map>
#include <Eigen/Sparse>

namespace RVO {
enum RVOOptimizer {
  NEWTON,
  LBFGS,
  UNKNOWN,
};
class VisibilityGraph;
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
  struct AgentTarget {
    Vec2T _target;
    T _maxVelocity;
    Mat2T _DVDP;
  };
#endif
  RVOSimulator(const RVOSimulator& other);
#ifndef SWIG
  RVOSimulator& operator=(const RVOSimulator& other);
#endif
  RVOSimulator(T d0=1,T gTol=1e-4,T coef=1,T timestep=1,int maxIter=1000,bool radixSort=false,bool useHash=true,const std::string& optimizer="NEWTON");
  virtual ~RVOSimulator() {}
  bool getUseHash() const;
  T getMaxRadius() const;
  void clearAgent();
  void clearObstacle();
  int getNrObstacle() const;
  int getNrAgent() const;
#ifndef SWIG
  Mat2XT& getAgentPositions();
  Mat2XT& getAgentVelocities();
  const Vec& getAgentRadius() const;
#endif
#ifdef SWIG
  std::vector<Eigen::Matrix<double,2,1>> getObstacle(int i) const;
  Eigen::Matrix<double,2,-1> getAgentPositions() const;
  Eigen::Matrix<double,2,-1> getAgentVelocities() const;
  Eigen::Matrix<double,2,1> getAgentPosition(int i) const;
  Eigen::Matrix<double,2,1> getAgentVelocity(int i) const;
  Eigen::Matrix<double,2,2> getAgentDVDP(int i) const;
  double getAgentRadius(int i) const;
  int addAgent(const Eigen::Matrix<double,2,1>& pos,const Eigen::Matrix<double,2,1>& vel,double rad);
  void setAgentPosition(int i,const Eigen::Matrix<double,2,1>& pos);
  void setAgentVelocity(int i,const Eigen::Matrix<double,2,1>& vel);
  void setAgentTarget(int i,const Eigen::Matrix<double,2,1>& target,T maxVelocity);
  int addObstacle(std::vector<Eigen::Matrix<double,2,1>> vss);
#else
  std::vector<Vec2T> getObstacle(int i) const;
  Mat2XT getAgentPositions() const;
  Mat2XT getAgentVelocities() const;
  Mat2XT getAgentDiffVelocities() const;
  Vec2T getAgentPosition(int i) const;
  Vec2T getAgentVelocity(int i) const;
  Mat2T getAgentDVDP(int i) const;
  T getAgentRadius(int i) const;
  int addAgent(const Vec2T& pos,const Vec2T& vel,T rad);
  void setAgentPosition(int i,const Vec2T& pos);
  void setAgentVelocity(int i,const Vec2T& vel);
  void setAgentTarget(int i,const Vec2T& target,T maxVelocity);
  int addObstacle(std::vector<Vec2T> vss);
  std::shared_ptr<VisibilityGraph> getVisibility() const;
#endif
  void buildVisibility(const RVOSimulator& ref);
  void buildVisibility();
  void clearVisibility();
  void setNewtonParameter(int maxIter,T gTol,T d0,T coef=1);
  void setLBFGSParameter(int nrCorrect=5);
  void setTimestep(T timestep);
  T timestep() const;
  virtual bool optimize(bool requireGrad,bool output);
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
 protected:
  static T clog(T d,T* D,T* DD,T d0,T coef);
  bool lineSearch(T E,const Vec& g,const Vec& d,T& alpha,Vec& newX,
                  std::function<bool(const Vec&,T&)> eval,T alphaMin) const;
  bool energy(VecCM prevPos,VecCM pos,T* f,Vec* g,SMatT* h,Eigen::Matrix<int,4,1>& nBarrier);
  bool energyAA(int aid,int bid,const Vec2T& a,const Vec2T& b,T* f,Vec* g,STrips* trips,Eigen::Matrix<int,4,1>& nBarrier) const;
  bool energyAO(int aid,const Vec2T& a,const Vec2T o[2],T* f,Vec* g,STrips* trips,Eigen::Matrix<int,4,1>& nBarrier) const;
  bool optimizeNewton(bool requireGrad,bool output);
  bool optimizeLBFGS(bool requireGrad,bool output);
  std::shared_ptr<VisibilityGraph> _vis;
  std::shared_ptr<SpatialHash> _hash;
  BoundingVolumeHierarchy _bvh;
  Mat2XT _perfVelocities;
  Mat2XT _agentPositions;
  Vec _agentRadius;
  Eigen::SimplicialLDLT<SMatT> _sol;
  std::unordered_map<int,AgentTarget> _agentTargets;
  T _timestep,_gTol,_d0,_coef,_maxRad;
  bool _useHash;
  int _maxIter;
  LBFGSUpdate _LBFGSUpdate;
  RVOOptimizer _optimizer;
  //data
  MatT _DXDX,_DXDV;
  SMatT _id;
#endif
};
}

#endif

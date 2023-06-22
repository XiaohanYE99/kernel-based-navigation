#ifndef RVO_H
#define RVO_H

#include "BoundingVolumeHierarchy.h"
#include "ParallelVector.h"
#include "DynamicMatVec.h"
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
  DECL_MAT_VEC_MAP_TYPES_I
  DECL_MAT_VEC_MAP_TYPES_T
  DECL_MAP_FUNCS
  typedef Eigen::Triplet<T,int> STrip;
  typedef ParallelVector<STrip> STrips;
  typedef Eigen::SparseMatrix<T,0,int> SMatT;
  struct AgentTarget {
    Vec2T _target;
    T _maxVelocity;
    Mat2T _DVDP;
  };
  RVOSimulator(const RVOSimulator& other);
  RVOSimulator& operator=(const RVOSimulator& other);
  RVOSimulator(T d0=1,T gTol=1e-4,T coef=1,T timestep=1,int maxIter=1000,bool radixSort=false,bool useHash=true,const std::string& optimizer="NEWTON");
  virtual ~RVOSimulator() {}
  bool getUseHash() const;
  T getMaxRadius() const;
  T getMinRadius() const;
  void clearAgent();
  void clearObstacle();
  int getNrObstacle() const;
  int getNrAgent() const;
  VecM getAgentPositionsVec();
  Mat2XTM getAgentPositions();
  Mat2XTM getAgentVelocities();
  Mat2XT getAgentTargets() const;
  VecCM getAgentRadius() const;
  VeciCM getAgentId() const;
  std::vector<Vec2T> getObstacle(int i) const;
  Mat2XT getAgentPositions() const;
  Mat2XT getAgentVelocities() const;
  Vec2T getAgentPosition(int i) const;
  Vec2T getAgentVelocity(int i) const;
  Mat2T getAgentDVDP(int i) const;
  T getAgentRadius(int i) const;
  int getAgentId(int i) const;
  void removeAgent(int i);
  int addAgent(const Vec2T& pos,const Vec2T& vel,T rad,int id=-1);
  void setAgentPosition(int i,const Vec2T& pos);
  void setAgentVelocity(int i,const Vec2T& vel);
  void setAgentTarget(int i,const Vec2T& target,T maxVelocity);
  int addObstacle(std::vector<Vec2T> vss);
  std::shared_ptr<VisibilityGraph> getVisibility() const;
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
  void debugNeighbor(T scale,T dscale=1e-3);
  void debugEnergy(T scale,T dscale=1);
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
  bool energy(VecCM prevPos,VecCM pos,T* f,Vec* g,SMatT* h,Vec4i& nBarrier);
  bool energyAA(int aid,int bid,const Vec2T& a,const Vec2T& b,T* f,Vec* g,STrips* trips,Vec4i& nBarrier) const;
  bool energyAO(int aid,const Vec2T& a,const Vec2T o[2],T* f,Vec* g,STrips* trips,Vec4i& nBarrier) const;
  bool optimizeNewton(bool requireGrad,bool output);
  bool optimizeLBFGS(bool requireGrad,bool output);
  void updateIdentity();
  std::shared_ptr<SpatialHash> _hash;
  BoundingVolumeHierarchy _bvh;
  Eigen::SimplicialLDLT<SMatT> _sol;
  T _timestep,_gTol,_d0,_coef,_maxRad,_minRad;
  bool _useHash;
  int _maxIter;
  LBFGSUpdate _LBFGSUpdate;
  RVOOptimizer _optimizer;
  SMatT _id;
  //entries of these data must be synchrnoized
  std::unordered_map<int,AgentTarget> _agentTargets;
  std::shared_ptr<VisibilityGraph> _vis;
  DynamicMat<T> _perfVelocities;
  DynamicMat<T> _agentPositions;
  DynamicVec<T> _agentRadius;
  DynamicVec<int> _agentId;
  //differentiable data
  MatT _DXDX,_DXDV;
};
}

#endif

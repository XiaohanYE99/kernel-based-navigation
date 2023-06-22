#include "RVO.h"
#include "Epsilon.h"
#include "Visibility.h"
#include "SpatialHashLinkedList.h"
#include "SpatialHashRadixSort.h"
#include <iostream>

namespace RVO {
RVOSimulator::RVOSimulator(const RVOSimulator& other) {
  operator=(other);
}
RVOSimulator& RVOSimulator::operator=(const RVOSimulator& other) {
  _timestep=other._timestep;
  _gTol=other._gTol;
  _d0=other._d0;
  _coef=other._coef;
  _maxRad=other._maxRad;
  _minRad=other._minRad;
  _useHash=other._useHash;
  _maxIter=other._maxIter;
  _LBFGSUpdate.nCorrect(other._LBFGSUpdate.nCorrect());
  _optimizer=other._optimizer;
  //clone agents
  if(std::dynamic_pointer_cast<SpatialHashRadixSort>(other._hash))
    _hash.reset(new SpatialHashRadixSort());
  else _hash.reset(new SpatialHashLinkedList());
  clearAgent();
  for(int i=0; i<other.getNrAgent(); i++)
    addAgent(other.getAgentPosition(i),other.getAgentVelocity(i),other.getAgentRadius(i));
  //clone obstacle
  clearObstacle();
  for(int i=0; i<other.getNrObstacle(); i++)
    addObstacle(other.getObstacle(i));
  return *this;
}
RVOSimulator::RVOSimulator(T d0,T gTol,T coef,T timestep,int maxIter,bool radixSort,bool useHash,const std::string& optimizer) {
  if(radixSort)
    _hash.reset(new SpatialHashRadixSort());
  else _hash.reset(new SpatialHashLinkedList());
  setNewtonParameter(maxIter,gTol,d0,coef);
  setLBFGSParameter(10);
  setTimestep(timestep);
  _useHash=useHash;
  _optimizer=optimizer=="NEWTON"?NEWTON:optimizer=="LBFGS"?LBFGS:UNKNOWN;
}
bool RVOSimulator::getUseHash() const {
  return _useHash;
}
RVOSimulator::T RVOSimulator::getMaxRadius() const {
  return _maxRad;
}
RVOSimulator::T RVOSimulator::getMinRadius() const {
  return _minRad;
}
void RVOSimulator::clearAgent() {
  _hash->clearVertex();
  _perfVelocities.reset();
  _agentPositions.reset();
  _agentRadius.reset();
  _agentId.reset();
  _agentTargets.clear();
  _maxRad=0;
  _minRad=0;
}
void RVOSimulator::clearObstacle() {
  _bvh.clearObstacle();
}
int RVOSimulator::getNrObstacle() const {
  return _bvh.getNrObstacle();
}
int RVOSimulator::getNrAgent() const {
  return _agentPositions.cols();
}
RVOSimulator::VecM RVOSimulator::getAgentPositionsVec() {
  return _agentPositions.getMapV();
}
RVOSimulator::Mat2XTM RVOSimulator::getAgentPositions() {
  return _agentPositions.getMap();
}
RVOSimulator::Mat2XTM RVOSimulator::getAgentVelocities() {
  return _perfVelocities.getMap();
}
RVOSimulator::Mat2XT RVOSimulator::getAgentTargets() const {
  Mat2XT targets=Mat2XT::Zero(2,_agentTargets.size());
  for(const auto& t:_agentTargets)
    targets.col(t.first)=t.second._target;
  return targets;
}
RVOSimulator::VecCM RVOSimulator::getAgentRadius() const {
  return _agentRadius.getCMap();
}
RVOSimulator::VeciCM RVOSimulator::getAgentId() const {
  return _agentId.getCMap();
}
std::vector<RVOSimulator::Vec2T> RVOSimulator::getObstacle(int i) const {
  return _bvh.getObstacle(i);
}
RVOSimulator::Mat2XT RVOSimulator::getAgentPositions() const {
  return _agentPositions.getCMap();
}
RVOSimulator::Mat2XT RVOSimulator::getAgentVelocities() const {
  return _perfVelocities.getCMap();
}
RVOSimulator::Vec2T RVOSimulator::getAgentPosition(int i) const {
  return _agentPositions.getCMap().col(i);
}
RVOSimulator::Vec2T RVOSimulator::getAgentVelocity(int i) const {
  return _perfVelocities.getCMap().col(i);
}
RVOSimulator::Mat2T RVOSimulator::getAgentDVDP(int i) const {
  if(_vis)
    return _vis->getAgentDVDP(i);
  else return _agentTargets.find(i)->second._DVDP;
}
RVOSimulator::T RVOSimulator::getAgentRadius(int i) const {
  return _agentRadius.getCMap()[i];
}
int RVOSimulator::getAgentId(int i) const {
  return _agentId.getCMap()[i];
}
void RVOSimulator::removeAgent(int i) {
  int last=_agentPositions.cols()-1;
  if(_vis)
    _vis->removeAgent(i,last);
  if(_agentTargets.find(last)!=_agentTargets.end()) {
    _agentTargets[i]=_agentTargets[last];
    _agentTargets.erase(last);
  }
  _hash->removeVertex();
  _perfVelocities.del(i);
  _agentPositions.del(i);
  _agentRadius.del(i);
  _agentId.del(i);
}
int RVOSimulator::addAgent(const Vec2T& pos,const Vec2T& vel,T rad,int id) {
  _hash->addVertex();
  _perfVelocities.add(vel);
  _agentPositions.add(pos);
  _agentRadius.add(rad);
  _agentId.add(id);
  _maxRad=_agentRadius.getCMap().maxCoeff();
  _minRad=_agentRadius.getCMap().minCoeff();
  return _agentPositions.cols()-1;
}
void RVOSimulator::setAgentPosition(int i,const Vec2T& pos) {
  _agentPositions.getMap().col(i)=pos;
}
void RVOSimulator::setAgentVelocity(int i,const Vec2T& vel) {
  _perfVelocities.getMap().col(i)=vel;
}
void RVOSimulator::setAgentTarget(int i,const Vec2T& target,T maxVelocity) {
  if(_vis)
    _vis->setAgentTarget(i,target,maxVelocity);
  else {
    _agentTargets[i]._target=target;
    _agentTargets[i]._maxVelocity=maxVelocity;
  }
}
int RVOSimulator::addObstacle(std::vector<Vec2T> vss) {
  _bvh.addObstacle(vss);
  return getNrObstacle()-1;
}
std::shared_ptr<VisibilityGraph> RVOSimulator::getVisibility() const {
  return _vis;
}
void RVOSimulator::buildVisibility(const RVOSimulator& ref) {
  _vis.reset(new VisibilityGraph(*this,*(ref._vis)));
}
void RVOSimulator::buildVisibility() {
  _vis.reset(new VisibilityGraph(*this));
}
void RVOSimulator::clearVisibility() {
  _vis=NULL;
}
void RVOSimulator::setNewtonParameter(int maxIter,T gTol,T d0,T coef) {
  _maxIter=maxIter;
  _gTol=gTol;
  _d0=d0;
  _coef=coef;
}
void RVOSimulator::setLBFGSParameter(int nrCorrect) {
  _LBFGSUpdate.nCorrect(nrCorrect);
}
void RVOSimulator::setTimestep(T timestep) {
  _timestep=timestep;
}
RVOSimulator::T RVOSimulator::timestep() const {
  return _timestep;
}
bool RVOSimulator::optimize(bool requireGrad,bool output) {
  if(_optimizer==NEWTON)
    return optimizeNewton(requireGrad,output);
  else if(_optimizer==LBFGS)
    return optimizeLBFGS(requireGrad,output);
  else {
    std::cout << "Unknown optimizer!" << std::endl;
    return false;
  }
}
void RVOSimulator::updateAgentTargets() {
  if(_vis)
    _vis->updateAgentTargets(*this);
  else {
    for(auto& target:_agentTargets) {
      _perfVelocities.getMap().col(target.first)=target.second._target-_agentPositions.getMap().col(target.first);
      T len=_perfVelocities.getMap().col(target.first).norm();
      if(len>target.second._maxVelocity) {
        T coef=target.second._maxVelocity/len;
        _perfVelocities.getMap().col(target.first)*=coef;
        target.second._DVDP=(_perfVelocities.getMap().col(target.first)*_perfVelocities.getMap().col(target.first).transpose()-Mat2T::Identity())*coef;
      } else {
        target.second._DVDP=-Mat2T::Identity();
      }
    }
  }
}
RVOSimulator::MatT RVOSimulator::getDXDX() const {
  return _DXDX;
}
RVOSimulator::MatT RVOSimulator::getDXDV() const {
  return _DXDV;
}
void RVOSimulator::debugNeighbor(T scale,T dscale) {
  std::cout << __FUNCTION__ << std::endl;
  while(true) {
    Vec prevPos=Vec::Random(_agentPositions.getMap().size())*scale;
    Vec pos=prevPos+Vec::Random(_agentPositions.getMap().size())*dscale;
    _hash->buildSpatialHash(mapCV(prevPos),mapCV(pos),_maxRad);
    std::vector<char> coll,collBF,collRef;
    std::vector<AgentNeighbor> AAss,AAssBF;
    std::vector<AgentObstacleNeighbor> AOss,AOssBF;
    //fast
    T margin=sqrt(_minRad*_minRad*4+_d0)-_minRad*2;
    _hash->detectSphereBroad(coll,_agentPositions,_agentRadius,margin);
    _hash->detectSphereBroad([&](AgentNeighbor n)->bool{
      OMP_CRITICAL_
      AAss.push_back(n);
      return true;
    },*_hash,margin);
    margin=sqrt(_minRad*_minRad+_d0)-_minRad;
    _hash->detectImplicitShape([&](AgentObstacleNeighbor n)->bool{
      OMP_CRITICAL_
      AOss.push_back(n);
      return true;
    },_bvh,margin);
    //slow
    margin=sqrt(_minRad*_minRad*4+_d0)-_minRad*2;
    _hash->detectSphereBroadBF(collBF,_agentPositions,_agentRadius,margin);
    _hash->detectSphereBroadBF([&](AgentNeighbor n)->bool{
      OMP_CRITICAL_
      AAssBF.push_back(n);
      return true;
    },*_hash,margin);
    margin=sqrt(_minRad*_minRad+_d0)-_minRad;
    _hash->detectImplicitShapeBF([&](AgentObstacleNeighbor n)->bool{
      OMP_CRITICAL_
      AOssBF.push_back(n);
      return true;
    },_bvh,margin);
    //compare
    ASSERT_MSG(coll==collBF,"coll!=collBF")
    ASSERT_MSGV(coll.size()==collBF.size(),"coll.size()=%lu!=collBF.size()=%lu",coll.size(),collBF.size())
    ASSERT_MSGV(AAss.size()==AAssBF.size(),"AAss.size()=%lu!=AAssBF.size()=%lu",AAss.size(),AAssBF.size())
    ASSERT_MSGV(AOss.size()==AOssBF.size(),"AOss.size()=%lu!=AOssBF.size()=%lu",AOss.size(),AOssBF.size())
    if(AAss.empty() || AOss.empty())
      continue;
    collRef.assign(coll.size(),false);
    std::sort(AAss.begin(),AAss.end());
    std::sort(AAssBF.begin(),AAssBF.end());
    for(int i=0; i<(int)AAss.size(); i++) {
      ASSERT_MSGV(AAss[i]==AAssBF[i],"AAss[%d]!=AAssBF[%d]",i,i)
    }
    std::sort(AOss.begin(),AOss.end());
    std::sort(AOssBF.begin(),AOssBF.end());
    for(int i=0; i<(int)AOss.size(); i++) {
      ASSERT_MSGV(AOss[i]==AOssBF[i],"AOss[%d]!=AOssBF[%d]",i,i)
    }
    break;
  }
}
void RVOSimulator::debugEnergy(T scale,T dscale) {
  std::cout << __FUNCTION__ << std::endl;
  DEFINE_NUMERIC_DELTA_T(T)
  while(true) {
    Vec prevPos=Vec::Random(_agentPositions.getMap().size())*scale;
    Vec pos=prevPos+Vec::Random(_agentPositions.getMap().size())*dscale,pos2;
    Vec dx=Vec::Random(_agentPositions.getMap().size());
    T f,f2;
    Vec g,g2;
    SMatT h;
    Vec4i nBarrier;
    if(!energy(mapCV(prevPos),mapCV(pos),&f,&g,&h,nBarrier))
      continue;
    if((nBarrier.array()==0).any())
      continue;
    std::cout << "nBarrier=[" << nBarrier.transpose() << "]" << std::endl;
    ASSERT(energy(mapCV(prevPos),mapCV(pos2=pos+dx*Delta),&f2,&g2,NULL,nBarrier))
    DEBUG_GRADIENT("f",g.dot(dx),g.dot(dx)-(f2-f)/Delta)
    DEBUG_GRADIENT("h",(h*dx).norm(),(h*dx-(g2-g)/Delta).norm())
    break;
  }
}
std::shared_ptr<SpatialHash> RVOSimulator::getHash() const {
  return _hash;
}
const BoundingVolumeHierarchy& RVOSimulator::getBVH() const {
  return _bvh;
}
void RVOSimulator::addBlock(Vec& g,int r,const Vec2T& blk) {
#ifdef FORCE_ADD_DOUBLE_PRECISION
  OMP_CRITICAL_
#else
  OMP_ATOMIC_
#endif
  g[r+0]+=blk[0];
#ifdef FORCE_ADD_DOUBLE_PRECISION
  OMP_CRITICAL_
#else
  OMP_ATOMIC_
#endif
  g[r+1]+=blk[1];
}
RVOSimulator::T RVOSimulator::absMax(const SMatT& h) {
  T val=0;
  for(int k=0; k<h.outerSize(); ++k)
    for(typename SMatT::InnerIterator it(h,k); it; ++it)
      val=fmax(val,fabs(it.value()));
  return val;
}
//helper
RVOSimulator::T RVOSimulator::clog(T d,T* D,T* DD,T d0,T coef) {
  if(d<=0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  } else if(d>d0) {
    if(D)
      *D=0;
    if(DD)
      *DD=0;
    return 0;
  }
  T valLog=log(d/d0);
  T valLogC=valLog*(d-d0);
  T relD=(d-d0)/d;
  if(D)
    *D=-(2*valLogC+(d-d0)*relD)*coef;
  if(DD)
    *DD=-(4*relD-relD*relD+2*valLog)*coef;
  return -valLogC*(d-d0)*coef;
}
bool RVOSimulator::lineSearch(T E,const Vec& g,const Vec& d,T& alpha,Vec& newX,
                              std::function<bool(const Vec&,T&)> eval,T alphaMin) const {
  //we want to find: x+alpha*d
  //double alphaMin=1e-6;	//this is global
  T alphaInc=1.1;
  T alphaDec=0.6;
  T c=0.1;
  T newE;
  Vec X,evalPt;
  while(alpha>alphaMin) {
    evalPt=newX+alpha*d;
    bool succ=eval(evalPt,newE);
    if(succ && newE<E+c*g.dot(d)*alpha) {
      newX.swap(evalPt);
      alpha*=alphaInc;
      break;
    } else {
      alpha*=alphaDec;
    }
  }
  return alpha>alphaMin;
}
bool RVOSimulator::energy(VecCM prevPos,VecCM pos,T* f,Vec* g,SMatT* h,Vec4i& nBarrier) {
#define PREVA(A) prevPos.data()?Vec2T(prevPos.template segment<2>(A->_id*2)):Vec2T::Zero()
#define CURRA(A) pos.template segment<2>(A->_id*2)
#define PREVO(O) O->_pos
#define CURRO(O) O->_next->_pos
  //initialize
  STrips trips;
  bool succ=true;
  VecCM x=_agentPositions.getCMapV();
  VecCM v=_perfVelocities.getCMapV();
  if(f)
    *f=(pos-(x+v*_timestep)).squaredNorm()/(2*_timestep*_timestep);
  if(g)
    *g=(pos-(x+v*_timestep))/(_timestep*_timestep);
  if(h)
    for(int i=0; i<x.size(); i++)
      trips.push_back(STrip(i,i,1/(_timestep*_timestep)));
  //update hash
  nBarrier.setZero();
  _hash->buildSpatialHash(prevPos,pos,_maxRad,_useHash);
  //inter-agent query
  T margin=sqrt(_minRad*_minRad*4+_d0)-_minRad*2;
  auto computeEnergyAA=[&](AgentNeighbor n)->bool{
    if(!succ)
      return false;
    //CCD check
    const Vec2T edgeA[2]= {PREVA(n._v[0]),CURRA(n._v[0])};
    const Vec2T edgeB[2]= {PREVA(n._v[1]),CURRA(n._v[1])};
    if(prevPos.data() && BoundingVolumeHierarchy::intersect(edgeA,edgeB))
      succ=false;
    else if(!energyAA(n._v[0]->_id,n._v[1]->_id,edgeA[1],edgeB[1],f,g,h?&trips:NULL,nBarrier))
      succ=false;
    return true;
  };
  if(_useHash)
    _hash->detectSphereBroad(computeEnergyAA,*_hash,margin);
  else _hash->detectSphereBroadBF(computeEnergyAA,*_hash,margin);
  //agent-obstacle query
  margin=sqrt(_minRad*_minRad+_d0)-_minRad;
  auto computeEnergyAO=[&](AgentObstacleNeighbor n)->bool{
    if(!succ)
      return false;
    //CCD check
    const Vec2T edgeA[2]= {PREVA(n._v),CURRA(n._v)};
    const Vec2T edgeB[2]= {PREVO(n._o),CURRO(n._o)};
    if(prevPos.data() && BoundingVolumeHierarchy::intersect(edgeA,edgeB))
      succ=false;
    else if(!energyAO(n._v->_id,edgeA[1],edgeB,f,g,h?&trips:NULL,nBarrier))
      succ=false;
    return true;
  };
  if(_useHash)
    _hash->detectImplicitShape(computeEnergyAO,_bvh,margin);
  else _hash->detectImplicitShapeBF(computeEnergyAO,_bvh,margin);
  //assemble
  if(h) {
    h->resize(pos.size(),pos.size());
    h->setFromTriplets(trips.begin(),trips.end());
  }
  return succ;
#undef PREVA
#undef CURRA
#undef PREVO
#undef CURRO
}
bool RVOSimulator::energyAA(int aid,int bid,const Vec2T& a,const Vec2T& b,T* f,Vec* g,STrips* trips,Vec4i& nBarrier) const {
  T sumRad=_agentRadius.getCMap()[aid]+_agentRadius.getCMap()[bid];
  aid*=2;
  bid*=2;
  Vec2T ab=a-b;
  T D=0,DD=0,margin=ab.squaredNorm()-sumRad*sumRad;
  if(margin<=0)
    return false;
  if(margin<_d0) {
    nBarrier[0]++;
    if(f)
#ifdef FORCE_ADD_DOUBLE_PRECISION
      OMP_CRITICAL_
#else
      OMP_ATOMIC_
#endif
      *f+=clog(margin,g?&D:NULL,trips?&DD:NULL,_d0,_coef);	//this can be infinite or nan
    if(g) {
      addBlock(*g,aid,D*2*ab);
      addBlock(*g,bid,-D*2*ab);
    }
    if(trips) {
      Mat2T diag=D*2*Mat2T::Identity()+DD*4*ab*ab.transpose();
      addBlock(*trips,aid,aid,diag);
      addBlock(*trips,bid,bid,diag);
      addBlock(*trips,aid,bid,-diag);
      addBlock(*trips,bid,aid,-diag.transpose());
    }
  }
  return true;
}
bool RVOSimulator::energyAO(int aid,const Vec2T& a,const Vec2T o[2],T* f,Vec* g,STrips* trips,Vec4i& nBarrier) const {
  T rad=_agentRadius.getCMap()[aid];
  aid*=2;
  T radSq=rad*rad,D=0,DD=0;
  Vec2T obsVec=o[1]-o[0],relPos0=o[0]-a,relPos1=o[1]-a;
  T lenSq=obsVec.squaredNorm(),s=(-relPos0.dot(obsVec))/lenSq;
  if(s<0) {
    T distSq0=relPos0.squaredNorm();
    if(distSq0<=radSq)
      return false;
    if(distSq0<radSq+_d0) {
      nBarrier[1]++;
      if(f)
#ifdef FORCE_ADD_DOUBLE_PRECISION
        OMP_CRITICAL_
#else
        OMP_ATOMIC_
#endif
        *f+=clog(distSq0-radSq,g?&D:NULL,trips?&DD:NULL,_d0,_coef);	//this can be infinite or nan
      if(g)
        addBlock(*g,aid,-D*2*relPos0);
      if(trips)
        addBlock(*trips,aid,aid,D*2*Mat2T::Identity()+DD*4*relPos0*relPos0.transpose());
    }
  } else if(s>1) {
    T distSq1=relPos1.squaredNorm();
    if(distSq1<=radSq)
      return false;
    if(distSq1<radSq+_d0) {
      nBarrier[2]++;
      if(f)
#ifdef FORCE_ADD_DOUBLE_PRECISION
        OMP_CRITICAL_
#else
        OMP_ATOMIC_
#endif
        *f+=clog(distSq1-radSq,g?&D:NULL,trips?&DD:NULL,_d0,_coef);	//this can be infinite or nan
      if(g)
        addBlock(*g,aid,-D*2*relPos1);
      if(trips)
        addBlock(*trips,aid,aid,D*2*Mat2T::Identity()+DD*4*relPos1*relPos1.transpose());
    }
  } else {
    Vec2T n(-obsVec[1],obsVec[0]);
    n/=sqrt(lenSq);
    T dist=relPos0.dot(n);
    T distSq=dist*dist;
    if(distSq<=radSq)
      return false;
    if(distSq<radSq+_d0) {
      nBarrier[3]++;
      if(f)
#ifdef FORCE_ADD_DOUBLE_PRECISION
        OMP_CRITICAL_
#else
        OMP_ATOMIC_
#endif
        *f+=clog(distSq-radSq,g?&D:NULL,trips?&DD:NULL,_d0,_coef);
      if(g)
        addBlock(*g,aid,-D*2*n*dist);
      if(trips)
        addBlock(*trips,aid,aid,(D*2+DD*4*dist*dist)*n*n.transpose());
    }
  }
  return true;
}
bool RVOSimulator::optimizeNewton(bool requireGrad,bool output) {
  if(_agentPositions.cols()==0)
    return true;
  updateIdentity();
  VecM XFrom=_agentPositions.getMapV();
  Vec X=XFrom,newX=XFrom;
  Vec g,g2;
  SMatT h;
  T E;
  int iter;
  T lastAlpha;
  T maxPerturbation=1e2;
  T minPerturbation=1e-6;
  T perturbation=1e0;
  T perturbationDec=0.7;
  T perturbationInc=10.0;
  T alpha=1,alphaMin=1e-10;
  Vec4i nBarrier;
  for(iter=0; iter<_maxIter && alpha>alphaMin && perturbation<maxPerturbation; iter++) {
    //always use spatial hash to compute obstacle neighbors, but only use spatial hash to compute agent neighbors
    bool succ=energy(mapCV<Vec>(NULL),mapCV(newX),&E,&g,&h,nBarrier);
    if(!succ)
      return false;
    if(g.cwiseAbs().maxCoeff()<_gTol) {
      if(output)
        std::cout << "Exit on gTol<" << _gTol << std::endl;
      break;
    }
    if(iter==0) {
      T hMax=absMax(h);
      maxPerturbation*=std::max<T>(1.0,hMax);
      minPerturbation*=std::max<T>(1.0,hMax);
      perturbation*=std::max<T>(1.0,hMax);
    }
    if(output)
      std::cout << "iter=" << iter
                << " alpha=" << alpha
                << " E=" << E
                << " gNormInf=" << g.cwiseAbs().maxCoeff()
                << " perturbation=" << perturbation
                << " minPertubation=" << minPerturbation <<std::endl;
    //outer-loop of line search and newton direction computation
    while(true) {
      //ensure hessian factorization is successful
      while(perturbation<maxPerturbation) {
        _sol.compute(h+_id*perturbation);
        if(_sol.info()==Eigen::Success) {
          perturbation=fmax(perturbation*perturbationDec,minPerturbation);
          break;
        } else {
          perturbation*=perturbationInc;
        }
      }
      if(perturbation>=maxPerturbation) {
        if(output)
          std::cout << "Exit on perturbation>=maxPerturbation" <<std::endl;
        break;
      }
      //line search
      lastAlpha=alpha;
      succ=lineSearch(E,g,-_sol.solve(g),alpha,newX,[&](const Vec& evalPt,T& E2)->bool {
        return energy(mapCV(X),mapCV(evalPt),&E2,NULL,NULL,nBarrier);
      },alphaMin);
      if(succ) {
        X=newX;
        perturbation=fmax(perturbation*perturbationDec,minPerturbation);
        break;
      }
      //probably we need more perturbation to h
      perturbation*=perturbationInc;
      alpha=lastAlpha;
      if(output)
        std::cout<<"Increase perturbation to "<<perturbation<<std::endl;
    }
  }
  if(requireGrad) {
    perturbation=0;
    while(true) {
      _sol.compute(h+_id*perturbation);
      if(_sol.info()==Eigen::Success)
        break;
      else {
        perturbation=fmax(minPerturbation,perturbation*perturbationInc);
        if(output)
          std::cout << "Singular configuration during backward pass!" << std::endl;
      }
    }
    _DXDX=_sol.solve(_id.toDense()*(1/(_timestep*_timestep)));
    _DXDV=_sol.solve(_id.toDense()*(1/_timestep));
  } else {
    _DXDX.setZero(0,0);
    _DXDV.setZero(0,0);
  }
  XFrom=X;
  return iter<_maxIter && alpha>alphaMin && perturbation<maxPerturbation;
}
bool RVOSimulator::optimizeLBFGS(bool requireGrad,bool output) {
  if(_agentPositions.cols()==0)
    return true;
  updateIdentity();
  VecM XFrom=_agentPositions.getMapV();
  Vec X=XFrom,pos=XFrom,posPrev,s,y;
  Vec g,g2,gPrev,d;
  SMatT h;
  T E;
  int iter;
  T alpha=1,alphaMin=1e-10;
  Vec4i nBarrier;
  _LBFGSUpdate.reset(XFrom.size());
  bool succ=energy(mapCV<Vec>(NULL),mapCV(pos),&E,&g,NULL,nBarrier);
  d=-g;
  alpha=1/std::max<T>(g.norm(),1e-8);
  for(iter=0; succ && iter<_maxIter && alpha>alphaMin; iter++) {
    posPrev=pos;
    gPrev=g;
    //update
    succ=lineSearch(E,g,d,alpha,pos,[&](const Vec& evalPt,T& E2)->bool {
      return energy(mapCV(X),mapCV(evalPt),&E2,&g2,NULL,nBarrier);
    },alphaMin);
    if(!succ)
      break;
    g.swap(g2);
    //update LBFGS
    if(_LBFGSUpdate.nCorrect()<=0)
      d=-g;
    else {
      s=pos-posPrev;
      y=g-gPrev;
      _LBFGSUpdate.update(mapCV(s),mapCV(y));
      _LBFGSUpdate.mulHv(mapCV(g),mapV(d));
      d*=-1;
    }
    if(output)
      std::cout << "iter=" << iter
                << " alpha=" << alpha
                << " E=" << E
                << " gNormInf=" << g.cwiseAbs().maxCoeff() <<std::endl;
    //termination
    if(g.cwiseAbs().maxCoeff()<_gTol) {
      if(output)
        std::cout << "Exit on gTol<" << _gTol << std::endl;
      break;
    }
  }
  if(requireGrad) {
    energy(mapCV<Vec>(NULL),mapCV(pos),&E,&g,&h,nBarrier);
    _sol.compute(h);    //the safety of such factorization is dubious
    _DXDX=_sol.solve(_id.toDense()*(1/(_timestep*_timestep)));
    _DXDV=_sol.solve(_id.toDense()*(1/_timestep));
  } else {
    _DXDX.setZero(0,0);
    _DXDV.setZero(0,0);
  }
  XFrom=pos;
  return iter<_maxIter && alpha>alphaMin && succ;
}
void RVOSimulator::updateIdentity() {
  if(_id.rows()==getNrAgent()*2)
    return;
  STrips trips;
  for(int i=0; i<getNrAgent()*2; i++)
    trips.push_back(STrip(i,i,1));
  _id.resize(getNrAgent()*2,getNrAgent()*2);
  _id.setFromTriplets(trips.begin(),trips.end());
}
}

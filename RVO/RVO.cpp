#include "RVO.h"
#include "Epsilon.h"
#include "SpatialHashLinkedList.h"
#include "SpatialHashRadixSort.h"
#include <iostream>

namespace RVO {
RVO::RVO(T rad,T d0,T gTol,T coef,T timestep,int maxIter,bool radixSort) {
  if(radixSort)
    _hash.reset(new SpatialHashRadixSort());
  else _hash.reset(new SpatialHashLinkedList());
  setNewtonParameter(maxIter,gTol,d0,coef);
  setTimestep(1);
  _rad=rad;
}
void RVO::clearAgent() {
  if(std::dynamic_pointer_cast<SpatialHashRadixSort>(_hash))
    _hash.reset(new SpatialHashRadixSort());
  else _hash.reset(new SpatialHashLinkedList());
  _perfVelocities.resize(2,0);
  _agentPositions.resize(2,0);
  _id.resize(0,0);
}
void RVO::clearObstacle() {
  _bvh.clearObstacle();
}
const RVO::Mat2XT& RVO::getAgent() const {
  return _agentPositions;
}
int RVO::addAgent(const Vec2T& pos,const Vec2T& vel) {
  _hash->addVertex(std::shared_ptr<Agent>(new Agent(_agentPositions.cols())));
  {
    Mat2XT perfVelocities=Mat2XT::Zero(2,_perfVelocities.cols()+1);
    perfVelocities.block(0,0,2,_perfVelocities.cols())=_perfVelocities;
    perfVelocities.col(_perfVelocities.cols())=vel;
    perfVelocities.swap(_perfVelocities);
  }
  {
    Mat2XT agentPositions=Mat2XT::Zero(2,_agentPositions.cols()+1);
    agentPositions.block(0,0,2,_agentPositions.cols())=_agentPositions;
    agentPositions.col(_agentPositions.cols())=pos;
    agentPositions.swap(_agentPositions);
  }
  {
    _id.resize(_agentPositions.size(),_agentPositions.size());
    for(int i=0; i<_id.rows(); i++)
      _id.coeffRef(i,i)=1;
  }
  return _agentPositions.cols()-1;
}
void RVO::setAgent(int i,const Vec2T& pos,const Vec2T& vel) {
  _agentPositions.col(i)=pos;
  _perfVelocities.col(i)=vel;
}
void RVO::addObstacle(const std::vector<Vec2T>& vss) {
  _bvh.addObstacle(vss);
}
void RVO::setNewtonParameter(int maxIter,T gTol,T d0,T coef) {
  _maxIter=maxIter;
  _gTol=gTol;
  _d0=d0;
  _coef=coef;
}
void RVO::setAgentRadius(T radius) {
  _rad=radius;
}
void RVO::setTimestep(T timestep) {
  _timestep=timestep;
}
RVO::T RVO::timestep() const {
  return _timestep;
}
bool RVO::optimize(MatT* DXDV,MatT* DXDX,bool output) {
  Vec XFrom=Eigen::Map<const Vec>(_agentPositions.data(),_agentPositions.size());
  Vec X=XFrom,newX=XFrom;
  Vec g,g2;
  SMatT h;
  T E;
  int iter;
  T maxPerturbation=1e2;
  T minPertubation=1e-9;
  T perturbation=1e0;
  T perturbationDec=0.8;
  T perturbationInc=2.0;
  T alpha=1,alphaMin=1e-10;
  Eigen::SimplicialLDLT<SMatT> sol;
  Eigen::Matrix<int,4,1> nBarrier;
  T lastAlpha;
  for(iter=0; iter<_maxIter && alpha>alphaMin && perturbation<maxPerturbation; iter++) {
    //always use spatial hash to compute obstacle neighbors, but only use spatial hash to compute agent neighbors
    bool succ=energy(X,newX,&E,&g,&h,nBarrier);
    if(!succ)
      return false;
    if(g.cwiseAbs().maxCoeff()<_gTol) {
      if(output)
        std::cout << "Exit on gTol<" << _gTol << std::endl;
      break;
    }
    if(iter==0) {
      T hMax=absMax(h);
      maxPerturbation*=std::max(1.0,hMax);
      minPertubation*=std::max(1.0,hMax);
      perturbation*=std::max(1.0,hMax);
    }
    if(output)
      std::cout << "iter=" << iter
                << " alpha=" << alpha
                << " E=" << E
                << " gNormInf=" << g.cwiseAbs().maxCoeff()
                << " perturbation=" << perturbation
                << " minPertubation=" << minPertubation <<std::endl;
    //outer-loop of line search and newton direction computation
    while(true) {
      //ensure hessian factorization is successful
      while(perturbation<maxPerturbation) {
        sol.compute(_id*perturbation+h);
        if(sol.info()==Eigen::Success) {
          perturbation=std::max(perturbation*perturbationDec,minPertubation);
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
      succ=lineSearch(E,g,-sol.solve(g),alpha,newX,[&](const Vec& evalPt,T& E2)->bool {
        return energy(newX,evalPt,&E2,NULL,NULL,nBarrier);
      },alphaMin);
      if(succ) {
        perturbation=std::max(perturbation*perturbationDec,minPertubation);
        break;
      }
      //probably we need more perturbation to h
      perturbation*=perturbationInc;
      alpha=lastAlpha;
      if(output)
        std::cout<<"Increase perturbation to "<<perturbation<<std::endl;
    }
  }
  if(DXDV || DXDX) {
    sol.compute(h);
    if(DXDV)
      *DXDV=sol.solve(_id.toDense()/_timestep);
    if(DXDX)
      *DXDX=sol.solve(_id.toDense()/(_timestep*_timestep));
  }
  return iter<_maxIter && alpha>alphaMin && perturbation<maxPerturbation;
}
void RVO::debugNeighbor(T scale) {
  while(true) {
    Vec prevPos=Vec::Random(_agentPositions.size())*scale;
    Vec pos=Vec::Random(_agentPositions.size())*scale;
    _hash->buildSpatialHash(mapCV(prevPos),mapCV(pos),_rad);
    std::vector<AgentNeighbor> AAss,AAssBF;
    std::vector<AgentObstacleNeighbor> AOss,AOssBF;
    //fast
    T margin=sqrt(_rad*_rad*4+_d0)-_rad*2;
    _hash->detectSphereBroad([&](AgentNeighbor n)->bool{
      AAss.push_back(n);
      return true;
    },*_hash,margin);
    margin=sqrt(_rad*_rad+_d0)-_rad;
    _hash->detectImplicitShape([&](AgentObstacleNeighbor n)->bool{
      AOss.push_back(n);
      return true;
    },_bvh,margin);
    //slow
    margin=sqrt(_rad*_rad*4+_d0)-_rad*2;
    _hash->detectSphereBroadBF([&](AgentNeighbor n)->bool{
      AAssBF.push_back(n);
      return true;
    },*_hash,margin);
    margin=sqrt(_rad*_rad+_d0)-_rad;
    _hash->detectImplicitShapeBF([&](AgentObstacleNeighbor n)->bool{
      AOssBF.push_back(n);
      return true;
    },_bvh,margin);
    //compare
    ASSERT_MSGV(AAss.size()==AAssBF.size(),"AAss.size()=%d!=AAssBF.size()=%d",AAss.size(),AAssBF.size())
    ASSERT_MSGV(AOss.size()==AOssBF.size(),"AOss.size()=%d!=AOssBF.size()=%d",AOss.size(),AOssBF.size())
    if(AAss.empty() || AOss.empty())
      continue;
    std::sort(AAss.begin(),AAss.end());
    std::sort(AAssBF.begin(),AAssBF.end());
    for(int i=0;i<(int)AAss.size();i++) {
      ASSERT(AAss[i]==AAssBF[i])
    }
    std::sort(AOss.begin(),AOss.end());
    std::sort(AOssBF.begin(),AOssBF.end());
    for(int i=0;i<(int)AOss.size();i++) {
      ASSERT(AOss[i]==AOssBF[i])
    }
    break;
  }
}
void RVO::debugEnergy(T scale,T dscale) {
  DEFINE_NUMERIC_DELTA_T(T)
  while(true) {
    Vec prevPos=Vec::Random(_agentPositions.size())*scale;
    Vec pos=prevPos+Vec::Random(_agentPositions.size())*dscale;
    Vec dx=Vec::Random(_agentPositions.size());
    T f,f2;
    Vec g,g2;
    SMatT h;
    Eigen::Matrix<int,4,1> nBarrier;
    if(!energy(prevPos,pos,&f,&g,&h,nBarrier))
      continue;
    if((nBarrier.array()==0).any())
      continue;
    ASSERT(energy(prevPos,pos+dx*Delta,&f2,&g2,NULL,nBarrier))
    DEBUG_GRADIENT("f",g.dot(dx),g.dot(dx)-(f2-f)/Delta)
    DEBUG_GRADIENT("h",(h*dx).norm(),(h*dx-(g2-g)/Delta).norm())
    break;
  }
}
//helper
RVO::T RVO::clog(T d,T* D,T* DD,T d0,T coef) {
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
bool RVO::lineSearch(T E,const Vec& g,const Vec& d,T& alpha,Vec& newX,
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
    if(!succ && newE<E+c*g.dot(d)*alpha) {
      newX.swap(evalPt);
      alpha*=alphaInc;
      break;
    } else {
      alpha*=alphaDec;
    }
  }
  return alpha>alphaMin;
}
bool RVO::energy(const Vec& prevPos,const Vec& pos,T* f,Vec* g,SMatT* h,Eigen::Matrix<int,4,1>& nBarrier) {
#define PREVA(A) prevPos.template segment<2>(A->_id*2)
#define CURRA(A) pos.template segment<2>(A->_id*2)
#define PREVO(O) O->_pos
#define CURRO(O) O->_next->_pos
  //update hash
  nBarrier.setZero();
  _hash->buildSpatialHash(mapCV(prevPos),mapCV(pos),_rad);
  //initialize
  STrips trips;
  bool succ=true;
  Eigen::Map<const Vec> x(_agentPositions.data(),_agentPositions.size());
  Eigen::Map<const Vec> v(_perfVelocities.data(),_perfVelocities.size());
  if(f)
    *f=(pos-(x+v*_timestep)).squaredNorm()/(2*_timestep*_timestep);
  if(g)
    *g=(pos-(x+v*_timestep))/(_timestep*_timestep);
  if(h)
    for(int i=0;i<x.size();i++)
      trips.push_back(STrip(i,i,1/(_timestep*_timestep)));
  //inter-agent query
  T margin=sqrt(_rad*_rad*4+_d0)-_rad*2;
  _hash->detectSphereBroad([&](AgentNeighbor n)->bool{
    if(!succ)
      return false;
    //CCD check
    const Vec2T edgeA[2]={PREVA(n._v[0]),CURRA(n._v[0])};
    const Vec2T edgeB[2]={PREVA(n._v[1]),CURRA(n._v[1])};
    if(intersect(edgeA,edgeB))
      succ=false;
    else if(!energyAA(n._v[0]->_id,n._v[1]->_id,edgeA[1],edgeB[1],f,g,h?&trips:NULL,nBarrier))
      succ=false;
    return true;
  },*_hash,margin);
  //agent-obstacle query
  margin=sqrt(_rad*_rad+_d0)-_rad;
  _hash->detectImplicitShape([&](AgentObstacleNeighbor n)->bool{
    if(!succ)
      return false;
    //CCD check
    const Vec2T edgeA[2]={PREVA(n._v),CURRA(n._v)};
    const Vec2T edgeB[2]={PREVO(n._o),CURRO(n._o)};
    if(intersect(edgeA,edgeB))
      succ=false;
    else if(!energyAO(n._v->_id,edgeA[1],edgeB,f,g,h?&trips:NULL,nBarrier))
      succ=false;
    return true;
  },_bvh,margin);
  //assemble
  if(h) {
    h->resize(pos.size(),pos.size());
    h->setFromTriplets(trips.begin(),trips.end());
  }
  return succ;
}
bool RVO::energyAA(int aid,int bid,const Vec2T& a,const Vec2T& b,T* f,Vec* g,STrips* trips,Eigen::Matrix<int,4,1>& nBarrier) const {
  aid*=2;
  bid*=2;
  Vec2T ab=a-b;
  T D=0,DD=0,margin=ab.squaredNorm();
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
bool RVO::energyAO(int aid,const Vec2T& a,const Vec2T o[2],T* f,Vec* g,STrips* trips,Eigen::Matrix<int,4,1>& nBarrier) const {
  aid*=2;
  T radSq=_rad*_rad,D=0,DD=0;
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
bool RVO::intersect(const Vec2T edgeA[2],const Vec2T edgeB[2]) const {
  //edgeA[0]+s*(edgeA[1]-edgeA[0])=edgeB[0]+t*(edgeB[1]-edgeB[0])
  Mat2T LHS;
  Vec2T RHS=edgeB[0]-edgeA[0];
  LHS.col(0)= (edgeA[1]-edgeA[0]);
  LHS.col(1)=-(edgeB[1]-edgeB[0]);
  if(LHS.determinant()<Epsilon<T>::defaultEps()) {
    return false;   //parallel line segment, doesn't matter
  } else {
    Vec2T st=LHS.inverse()*RHS;
    return st[0]>=0 && st[0]<=1 && st[1]>=0 && st[1]<=1;
  }
}
void RVO::addBlock(Vec& g,int r,const Vec2T& blk) {
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
template <typename MAT>
void RVO::addBlock(STrips& trips,int r,int c,const MAT& blk) {
  for(int R=0; R<blk.rows(); R++)
    for(int C=0; C<blk.cols(); C++)
      trips.push_back(STrip(r+R,c+C,blk(R,C)));
}
RVO::T RVO::absMax(const SMatT& h) {
  T val=0;
  for(int k=0; k<h.outerSize(); ++k)
    for(typename SMatT::InnerIterator it(h,k); it; ++it)
      val=fmax(val,fabs(it.value()));
  return val;
}
}

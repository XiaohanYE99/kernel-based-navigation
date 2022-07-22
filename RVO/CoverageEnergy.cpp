#include "CoverageEnergy.h"
#include "SpatialHashLinkedList.h"
#include "SpatialHashRadixSort.h"
#include <iostream>

namespace RVO {
CoverageEnergy::CoverageEnergy(const RVOSimulator& sim,T range,bool visibleOnly)
  :_hash(sim.getHash()),_bvh(sim.getBVH()),_visibleOnly(visibleOnly),_useHash(sim.getUseHash()),_range(range) {}
CoverageEnergy::T CoverageEnergy::loss(Vec pos) {
  T f=0;
  Eigen::Matrix<int,4,1> nBarrier;
  energy(mapCV(pos),&f,&_grad,(STrips*)NULL,nBarrier);
  return f;
}
CoverageEnergy::Vec CoverageEnergy::grad() const {
  return _grad;
}
void CoverageEnergy::debugCoverage(T scale) {
  std::cout << __FUNCTION__ << " " << (_visibleOnly?"VisibleOnly":"Non-VisibleOnly") << std::endl;
  DEFINE_NUMERIC_DELTA_T(T)
  while(true) {
    Vec pos=Vec::Random((int)_hash->vss().size()*2)*scale,pos2;
    Vec dx=Vec::Random((int)_hash->vss().size()*2);
    T f,f2;
    Vec g,g2;
    SMatT h;
    Eigen::Matrix<int,4,1> nBarrier;
    energy(mapCV(pos),&f,&g,&h,nBarrier);
    if((nBarrier.array()==0).any())
      continue;
    std::cout << "nBarrier=[" << nBarrier.transpose() << "]" << std::endl;
    energy(mapCV(pos2=pos+dx*Delta),&f2,&g2,(SMatT*)NULL,nBarrier);
    DEBUG_GRADIENT("f",g.dot(dx),g.dot(dx)-(f2-f)/Delta)
    DEBUG_GRADIENT("h",(h*dx).norm(),(h*dx-(g2-g)/Delta).norm())
    break;
  }
}
void CoverageEnergy::energy(VecCM pos,T* f,Vec* g,STrips* trips,Eigen::Matrix<int,4,1>& nBarrier) {
#define CURRA(A) pos.template segment<2>(A->_id*2)
#define PREVO(O) O->_pos
#define CURRO(O) O->_next->_pos
  //initialize
  if(f)
    *f=0;
  if(g)
    g->setZero(pos.size());
  if(trips)
    trips->clear();
  //update hash
  nBarrier.setZero();
  _hash->buildSpatialHash(pos,mapCV<Vec>(NULL),_range,_useHash);
  //inter-agent query
  auto computeEnergyAA=[&](AgentNeighbor n)->bool{
    const Vec2T posA=CURRA(n._v[0]);
    const Vec2T posB=CURRA(n._v[1]);
    if(_visibleOnly && !_bvh.visible(posA,posB))
      return true;
    energyAA(n._v[0]->_id,n._v[1]->_id,posA,posB,f,g,trips,nBarrier);
    return true;
  };
  if(_useHash)
    _hash->detectSphereBroad(computeEnergyAA,*_hash,0);
  else _hash->detectSphereBroadBF(computeEnergyAA,*_hash,0);
  //agent-obstacle query
  auto computeEnergyAO=[&](AgentObstacleNeighbor n)->bool{
    const Vec2T posA=CURRA(n._v);
    if(_visibleOnly && !_bvh.visible(posA,n._o))
      return true;
    const Vec2T edgeB[2]= {PREVO(n._o),CURRO(n._o)};
    energyAO(n._v->_id,posA,edgeB,f,g,trips,nBarrier);
    return true;
  };
  if(_useHash)
    _hash->detectImplicitShape(computeEnergyAO,_bvh,0);
  else _hash->detectImplicitShapeBF(computeEnergyAO,_bvh,0);
#undef PREVA
#undef CURRA
#undef PREVO
#undef CURRO
}
void CoverageEnergy::energy(VecCM pos,T* f,Vec* g,SMatT* h,Eigen::Matrix<int,4,1>& nBarrier) {
  STrips trips;
  energy(pos,f,g,h?&trips:NULL,nBarrier);
  if(h) {
    h->resize(pos.size(),pos.size());
    h->setFromTriplets(trips.begin(),trips.end());
  }
}
//helper
void CoverageEnergy::energyAA(int aid,int bid,const Vec2T& a,const Vec2T& b,T* f,Vec* g,STrips* trips,Eigen::Matrix<int,4,1>& nBarrier) const {
  aid*=2;
  bid*=2;
  Vec2T ab=a-b;
  T D=0,DD=0,margin=ab.squaredNorm();
  if(margin<=_range*_range) {
    if(f)
#ifdef FORCE_ADD_DOUBLE_PRECISION
      OMP_CRITICAL_
#else
      OMP_ATOMIC_
#endif
      (*f)+=kernel(margin,g?&D:NULL,trips?&DD:NULL,_range);
    if(g) {
      RVOSimulator::addBlock(*g,aid,D*2*ab);
      RVOSimulator::addBlock(*g,bid,-D*2*ab);
    }
    if(trips) {
      Mat2T diag=D*2*Mat2T::Identity()+DD*4*ab*ab.transpose();
      RVOSimulator::addBlock(*trips,aid,aid,diag);
      RVOSimulator::addBlock(*trips,bid,bid,diag);
      RVOSimulator::addBlock(*trips,aid,bid,-diag);
      RVOSimulator::addBlock(*trips,bid,aid,-diag.transpose());
    }
    nBarrier[0]++;
  }
}
void CoverageEnergy::energyAO(int aid,const Vec2T& a,const Vec2T o[2],T* f,Vec* g,STrips* trips,Eigen::Matrix<int,4,1>& nBarrier) const {
  aid*=2;
  T D=0,DD=0;
  Vec2T obsVec=o[1]-o[0],relPos0=o[0]-a,relPos1=o[1]-a;
  T lenSq=obsVec.squaredNorm(),s=(-relPos0.dot(obsVec))/lenSq;
  if(s<0) {
    T distSq0=relPos0.squaredNorm();
    if(distSq0<_range*_range) {
      if(f)
#ifdef FORCE_ADD_DOUBLE_PRECISION
        OMP_CRITICAL_
#else
        OMP_ATOMIC_
#endif
        *f+=kernel(distSq0,g?&D:NULL,trips?&DD:NULL,_range);	//this can be infinite or nan
      if(g)
        RVOSimulator::addBlock(*g,aid,-D*2*relPos0);
      if(trips)
        RVOSimulator::addBlock(*trips,aid,aid,D*2*Mat2T::Identity()+DD*4*relPos0*relPos0.transpose());
      nBarrier[1]++;
    }
  } else if(s>1) {
    T distSq1=relPos1.squaredNorm();
    if(distSq1<_range*_range) {
      if(f)
#ifdef FORCE_ADD_DOUBLE_PRECISION
        OMP_CRITICAL_
#else
        OMP_ATOMIC_
#endif
        *f+=kernel(distSq1,g?&D:NULL,trips?&DD:NULL,_range);	//this can be infinite or nan
      if(g)
        RVOSimulator::addBlock(*g,aid,-D*2*relPos1);
      if(trips)
        RVOSimulator::addBlock(*trips,aid,aid,D*2*Mat2T::Identity()+DD*4*relPos1*relPos1.transpose());
      nBarrier[2]++;
    }
  } else {
    Vec2T n(-obsVec[1],obsVec[0]);
    n/=sqrt(lenSq);
    T dist=relPos0.dot(n);
    T distSq=dist*dist;
    if(distSq<_range*_range) {
      if(f)
#ifdef FORCE_ADD_DOUBLE_PRECISION
        OMP_CRITICAL_
#else
        OMP_ATOMIC_
#endif
        *f+=kernel(distSq,g?&D:NULL,trips?&DD:NULL,_range);
      if(g)
        RVOSimulator::addBlock(*g,aid,-D*2*n*dist);
      if(trips)
        RVOSimulator::addBlock(*trips,aid,aid,(D*2+DD*4*dist*dist)*n*n.transpose());
      nBarrier[3]++;
    }
  }
}
CoverageEnergy::T CoverageEnergy::kernel(T distSq,T* D,T* DD,T range) {
  T len=sqrt(distSq);
  T q=len*2/range;
  if(q>=2) {
    if(D)
      *D=0;
    if(DD)
      *DD=0;
    return 0;
  }
  T a=1-q/2,a2=a*a,a3=a2*a;
  T b=1.5*q+1;
  if(D) {
    *D=(3*a3-(3*a2*b))/(2*range*len);
  }
  if(DD) {
    *DD=(3*a*b-9*a2)/(2*pow(range*len,2));
    *DD-=(3*a3-3*a2*b)/(4*range*len*distSq);
  }
  return a3*b;
}
}

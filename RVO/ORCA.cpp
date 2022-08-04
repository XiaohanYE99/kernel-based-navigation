#include "ORCA.h"
#include "Epsilon.h"
#include "BoundingVolumeHierarchy.h"
#include "SpatialHashLinkedList.h"
#include "SpatialHashRadixSort.h"
#include <iostream>

namespace RVO {
//VelocityObstacle
VelocityObstacle::Vec2T VelocityObstacle::proj(const Vec2T& v,T tol) const {
  return v-((v-pos()).dot(nor())-tol)*nor();
}
VelocityObstacle::Vec2T VelocityObstacle::proj(const Vec2T& v) const {
  return v-(v-pos()).dot(nor())*nor();
}
VelocityObstacle::T VelocityObstacle::violation(const Vec2T& v) const {
  return fmax((T)0,-(v-pos()).dot(nor()));
}
bool VelocityObstacle::outside(const Vec2T& v) const {
  return (v-pos()).dot(nor())>0;
}
VelocityObstacle::Vec2T VelocityObstacle::pos() const {
  return Vec2T(_pos[0].value(),_pos[1].value());
}
VelocityObstacle::Vec2T VelocityObstacle::nor() const {
  return Vec2T(_nor[0].value(),_nor[1].value());
}
VelocityObstacle::Mat2T VelocityObstacle::DposDpa() const {
  Mat2T ret;
  ASSERT(_aid>=0)
  ret.row(0)=_pos[0].derivatives().template segment<2>(0);
  ret.row(1)=_pos[1].derivatives().template segment<2>(0);
  return ret;
}
VelocityObstacle::Mat2T VelocityObstacle::DposDpb() const {
  Mat2T ret;
  ASSERT(_bid>=0)
  ret.row(0)=_pos[0].derivatives().template segment<2>(2);
  ret.row(1)=_pos[1].derivatives().template segment<2>(2);
  return ret;
}
VelocityObstacle::Mat2T VelocityObstacle::DposDva() const {
  Mat2T ret;
  ASSERT(_aid>=0)
  ret.row(0)=_pos[0].derivatives().template segment<2>(4);
  ret.row(1)=_pos[1].derivatives().template segment<2>(4);
  return ret;
}
VelocityObstacle::Mat2T VelocityObstacle::DposDvb() const {
  Mat2T ret;
  ASSERT(_bid>=0)
  ret.row(0)=_pos[0].derivatives().template segment<2>(6);
  ret.row(1)=_pos[1].derivatives().template segment<2>(6);
  return ret;
}
VelocityObstacle::Mat2T VelocityObstacle::DnorDpa() const {
  Mat2T ret;
  ASSERT(_aid>=0)
  ret.row(0)=_nor[0].derivatives().template segment<2>(0);
  ret.row(1)=_nor[1].derivatives().template segment<2>(0);
  return ret;
}
VelocityObstacle::Mat2T VelocityObstacle::DnorDpb() const {
  Mat2T ret;
  ASSERT(_bid>=0)
  ret.row(0)=_nor[0].derivatives().template segment<2>(2);
  ret.row(1)=_nor[1].derivatives().template segment<2>(2);
  return ret;
}
VelocityObstacle::Mat2T VelocityObstacle::DnorDva() const {
  Mat2T ret;
  ASSERT(_aid>=0)
  ret.row(0)=_nor[0].derivatives().template segment<2>(4);
  ret.row(1)=_nor[1].derivatives().template segment<2>(4);
  return ret;
}
VelocityObstacle::Mat2T VelocityObstacle::DnorDvb() const {
  Mat2T ret;
  ASSERT(_bid>=0)
  ret.row(0)=_nor[0].derivatives().template segment<2>(6);
  ret.row(1)=_nor[1].derivatives().template segment<2>(6);
  return ret;
}
//ORCASimulator
ORCASimulator::ORCASimulator(const ORCASimulator& other):RVOSimulator(other) {}
ORCASimulator& ORCASimulator::operator=(const ORCASimulator& other) {
  RVOSimulator::operator=(other);
  return *this;
}
ORCASimulator::ORCASimulator(T rad,T d0,T gTol,T coef,T timestep,int maxIter,bool radixSort,bool useHash)
  :RVOSimulator(rad,d0,gTol,coef,timestep,maxIter,radixSort,useHash) {}
bool ORCASimulator::optimize(bool requireGrad,bool output) {
  //initialize
  _LPs.assign(_agentPositions.cols(), {});
  if(requireGrad) {
    _DXDX.setZero(_agentPositions.size(),_agentPositions.size());
    _DXDV.setZero(_agentPositions.size(),_agentPositions.size());
  }
  std::vector<omp_lock_t> locks(_agentPositions.cols());
  OMP_PARALLEL_FOR_
  for(int i=0; i<_agentPositions.cols(); i++)
    omp_init_lock(&locks[i]);
  //Stage 1: build hash
  //we need to build a hash spanning the largest influence range: v*dt+r
  if(output)
    std::cout << "Computing agent hash!" << std::endl;
  Vec prevPositions=Vec::Zero(_agentPositions.size());
  Vec nextPositions=Vec::Zero(_agentPositions.size());
  Eigen::Map<Mat2XT>(prevPositions.data(),2,_agentPositions.cols())=_agentPositions-_perfVelocities*_timestep;
  Eigen::Map<Mat2XT>(nextPositions.data(),2,_agentPositions.cols())=_agentPositions+_perfVelocities*_timestep;
  _hash->buildSpatialHash(mapCV(prevPositions),mapCV(nextPositions),_rad);
  Mat2XT tmpPerfVelocity=_perfVelocities;
  _perfVelocities.setZero();    //we need to set perfered velocity to zero, ensuring feasibility
  //Stage 2: compute velocity obstacles between all pairs of agents
  if(output)
    std::cout << "Computing agent velocity obstacles!" << std::endl;
  auto computeVAFunc=[&](AgentNeighbor n)->bool {
    for(int i:{0,1}) {
      VelocityObstacle VO=computeVelocityObstacle(n._v[i]->_id,n._v[1-i]->_id);
      omp_set_lock(&locks[VO._aid]);
      _LPs[VO._aid].push_back(VO);
      omp_unset_lock(&locks[VO._aid]);
    }
    return true;
  };
  if(_useHash)
    _hash->detectSphereBroad(computeVAFunc,*_hash,0);
  else _hash->detectSphereBroadBF(computeVAFunc,*_hash,0);
  //Stage 3: compute velocity obstacles between all pairs of agent-and-obstacles
  if(output)
    std::cout << "Computing agent-obstacle velocity obstacles!" << std::endl;
  auto computeVOFunc=[&](AgentObstacleNeighbor n)->bool {
    Vec2T obs[2]= {n._o->_pos,n._o->_next->_pos};
    VelocityObstacle VO=computeVelocityObstacle(n._v->_id,obs);
    omp_set_lock(&locks[VO._aid]);
    _LPs[VO._aid].push_back(VO);
    omp_unset_lock(&locks[VO._aid]);
    return true;
  };
  if(_useHash)
    _hash->detectImplicitShape(computeVOFunc,_bvh,0);
  else _hash->detectImplicitShapeBF(computeVOFunc,_bvh,0);
  //Stage 4: solve linear programming
  bool succ=true;
  if(output)
    std::cout << "Adjusting velocities!" << std::endl;
  _LPSolutions.resize(_agentPositions.cols());
  OMP_PARALLEL_FOR_
  for(int i=0; i<_agentPositions.cols(); i++) {
    _LPSolutions[i]=solveLP(tmpPerfVelocity.col(i),_LPs[i],_gTol);
    _perfVelocities.col(i)=_LPSolutions[i]._vOut;
    _agentPositions.col(i)+=_perfVelocities.col(i)*_timestep;
    if(!_LPSolutions[i]._succ)
      succ=false;
    //Stage 5: compute DVDX, DVDV
    if(requireGrad)
      buildGrad(i,_DXDX,_DXDV,_LPSolutions[i],_LPs[i],_gTol);
    omp_destroy_lock(&locks[i]);
  }
  //Stage 6: compute DXDX, DXDV
  if(requireGrad) {
    _DXDX*=_timestep;
    _DXDV*=_timestep;
    _DXDX+=MatT::Identity(_DXDX.rows(),_DXDX.cols());
  }
  return succ;
}
void ORCASimulator::debugVO(int aid,int bid,int testCase,T eps) {
  if(aid<0)
    do {
      aid=rand()%getNrAgent();
    } while(aid==bid);
  if(bid<0)
    do {
      bid=rand()%getNrAgent();
    } while(aid==bid);
  DEFINE_NUMERIC_DELTA_T(T)
  std::cout << "Debugging VO-Agent Case " << testCase << std::endl;
  //test the projection functionality
  while(true) {
    _agentPositions.setRandom();
    _perfVelocities.setRandom();
    if((_agentPositions.col(aid)-_agentPositions.col(bid)).norm()<=_rad*2)
      continue;
    VelocityObstacle VO0=computeVelocityObstacle(aid,bid);
    VelocityObstacle VO1=computeVelocityObstacle(bid,aid);
    debugDerivatives(VO0);
    debugDerivatives(VO1);
    ASSERT_MSGV(VO0._case==VO1._case,"%d=VO0._case!=VO1._case=%d!",VO0._case,VO1._case)
    if(VO0._case!=testCase)
      continue;
    _perfVelocities.col(aid)=VO0.proj(_perfVelocities.col(aid));
    _perfVelocities.col(bid)=VO1.proj(_perfVelocities.col(bid));
    Vec2T edgeA[2]= {_agentPositions.col(aid),_agentPositions.col(aid)+_perfVelocities.col(aid)*_timestep};
    Vec2T edgeB[2]= {_agentPositions.col(bid),_agentPositions.col(bid)+_perfVelocities.col(bid)*_timestep};
    T dist=BoundingVolumeHierarchy::distanceAgentAgent(edgeA,edgeB);
    DEBUG_GRADIENT("VO",_rad*2,_rad*2-dist)
    if(abs(_rad*2-dist)>eps) {
      std::cout << "error too large!" << std::endl;
      exit(EXIT_FAILURE);
    }
    break;
  }
  //test the side of constraint: we set the agents to be collision-free, the constraints should be non-violated
  while(true) {
    _agentPositions.setRandom();
    _perfVelocities.setRandom();
    if((_agentPositions.col(aid)-_agentPositions.col(bid)).norm()<=_rad*2)
      continue;
    VelocityObstacle VO0=computeVelocityObstacle(aid,bid);
    VelocityObstacle VO1=computeVelocityObstacle(bid,aid);
    ASSERT_MSGV(VO0._case==VO1._case,"%d=VO0._case!=VO1._case=%d!",VO0._case,VO1._case)
    if(VO0._case!=testCase)
      continue;
    Vec2T edgeA[2]= {_agentPositions.col(aid),_agentPositions.col(aid)+_perfVelocities.col(aid)*_timestep};
    Vec2T edgeB[2]= {_agentPositions.col(bid),_agentPositions.col(bid)+_perfVelocities.col(bid)*_timestep};
    T dist=BoundingVolumeHierarchy::distanceAgentAgent(edgeA,edgeB);
    if(dist<=_rad*2)
      continue;
    ASSERT_MSG(VO0.outside(_perfVelocities.col(aid)),"VO0 not outside!")
    ASSERT_MSG(VO1.outside(_perfVelocities.col(bid)),"VO1 not outside!")
    break;
  }
  //test the side of constraint: we set the agents to be colliding, the constraints should be violated
  while(true) {
    _agentPositions.setRandom();
    _perfVelocities.setRandom();
    if((_agentPositions.col(aid)-_agentPositions.col(bid)).norm()<=_rad*2)
      continue;
    VelocityObstacle VO0=computeVelocityObstacle(aid,bid);
    VelocityObstacle VO1=computeVelocityObstacle(bid,aid);
    ASSERT_MSGV(VO0._case==VO1._case,"%d=VO0._case!=VO1._case=%d!",VO0._case,VO1._case)
    if(VO0._case!=testCase)
      continue;
    Vec2T edgeA[2]= {_agentPositions.col(aid),_agentPositions.col(aid)+_perfVelocities.col(aid)*_timestep};
    Vec2T edgeB[2]= {_agentPositions.col(bid),_agentPositions.col(bid)+_perfVelocities.col(bid)*_timestep};
    T dist=BoundingVolumeHierarchy::distanceAgentAgent(edgeA,edgeB);
    if(dist>=_rad*2)
      continue;
    ASSERT_MSG(!VO0.outside(_perfVelocities.col(aid)),"VO0 not inside!")
    ASSERT_MSG(!VO1.outside(_perfVelocities.col(bid)),"VO1 not inside!")
    break;
  }
}
void ORCASimulator::debugVO(int aid,int testCase,bool mustInside,T eps) {
  if(aid<0)
    aid=rand()%getNrAgent();
  DEFINE_NUMERIC_DELTA_T(T)
  std::cout << "Debugging VO-Obstacle Case " << testCase << std::endl;
  //test the projection functionality
  while(true) {
    _agentPositions.setRandom();
    _perfVelocities.setRandom();
    Vec2T o[2]= {Vec2T::Random(),Vec2T::Random()};
    if(BoundingVolumeHierarchy::distance(_agentPositions.col(aid),o)<=_rad)
      continue;
    VelocityObstacle VO=computeVelocityObstacle(aid,o);
    debugDerivatives(VO,o);
    if(VO._case!=testCase)
      continue;
    _perfVelocities.col(aid)=VO.proj(_perfVelocities.col(aid));
    Vec2T edgeA[2]= {_agentPositions.col(aid),_agentPositions.col(aid)+_perfVelocities.col(aid)*_timestep};
    T dist=BoundingVolumeHierarchy::distanceAgentObstacle(edgeA,o);
    DEBUG_GRADIENT("VO",_rad,_rad-dist)
    if(abs(_rad-dist)>eps) {
      std::cout << "error too large!" << std::endl;
      exit(EXIT_FAILURE);
    }
    break;
  }
  if(mustInside)
    return;
  //test the side of constraint: we set the agents to be collision-free, the constraints should be non-violated
  while(true) {
    _agentPositions.setRandom();
    _perfVelocities.setRandom();
    Vec2T o[2]= {Vec2T::Random(),Vec2T::Random()};
    if(BoundingVolumeHierarchy::distance(_agentPositions.col(aid),o)<=_rad)
      continue;
    VelocityObstacle VO=computeVelocityObstacle(aid,o);
    if(VO._case!=testCase)
      continue;
    Vec2T edgeA[2]= {_agentPositions.col(aid),_agentPositions.col(aid)+_perfVelocities.col(aid)*_timestep};
    T dist=BoundingVolumeHierarchy::distanceAgentObstacle(edgeA,o);
    if(dist<=_rad)
      continue;
    ASSERT_MSG(VO.outside(_perfVelocities.col(aid)),"VO not outside!")
    break;
  }
  //test the side of constraint: we set the agents to be colliding, the constraints should be violated
  while(true) {
    _agentPositions.setRandom();
    _perfVelocities.setRandom();
    Vec2T o[2]= {Vec2T::Random(),Vec2T::Random()};
    if(BoundingVolumeHierarchy::distance(_agentPositions.col(aid),o)<=_rad)
      continue;
    VelocityObstacle VO=computeVelocityObstacle(aid,o);
    if(VO._case!=testCase)
      continue;
    Vec2T edgeA[2]= {_agentPositions.col(aid),_agentPositions.col(aid)+_perfVelocities.col(aid)*_timestep};
    T dist=BoundingVolumeHierarchy::distanceAgentObstacle(edgeA,o);
    if(dist>=_rad)
      continue;
    ASSERT_MSG(!VO.outside(_perfVelocities.col(aid)),"VO not inside!")
    break;
  }
}
void ORCASimulator::debugDerivatives(const VelocityObstacle& VO) {
  Vec2T d=Vec2T::Random(),old;
  VelocityObstacle VO2;
  DEFINE_NUMERIC_DELTA_T(T)
  //pa
  old=_agentPositions.col(VO._aid);
  _agentPositions.col(VO._aid)+=d*Delta;
  VO2=computeVelocityObstacle(VO._aid,VO._bid);
  DEBUG_GRADIENT("DposDpa",(VO.DposDpa()*d).norm(),(VO.DposDpa()*d-(VO2.pos()-VO.pos())/Delta).norm())
  DEBUG_GRADIENT("DnorDpa",(VO.DnorDpa()*d).norm(),(VO.DnorDpa()*d-(VO2.nor()-VO.nor())/Delta).norm())
  _agentPositions.col(VO._aid)=old;
  //va
  old=_perfVelocities.col(VO._aid);
  _perfVelocities.col(VO._aid)+=d*Delta;
  VO2=computeVelocityObstacle(VO._aid,VO._bid);
  DEBUG_GRADIENT("DposDva",(VO.DposDva()*d).norm(),(VO.DposDva()*d-(VO2.pos()-VO.pos())/Delta).norm())
  DEBUG_GRADIENT("DnorDva",(VO.DnorDva()*d).norm(),(VO.DnorDva()*d-(VO2.nor()-VO.nor())/Delta).norm())
  _perfVelocities.col(VO._aid)=old;
  //pb
  old=_agentPositions.col(VO._bid);
  _agentPositions.col(VO._bid)+=d*Delta;
  VO2=computeVelocityObstacle(VO._aid,VO._bid);
  DEBUG_GRADIENT("DposDpb",(VO.DposDpb()*d).norm(),(VO.DposDpb()*d-(VO2.pos()-VO.pos())/Delta).norm())
  DEBUG_GRADIENT("DnorDpb",(VO.DnorDpb()*d).norm(),(VO.DnorDpb()*d-(VO2.nor()-VO.nor())/Delta).norm())
  _agentPositions.col(VO._bid)=old;
  //vb
  old=_perfVelocities.col(VO._bid);
  _perfVelocities.col(VO._bid)+=d*Delta;
  VO2=computeVelocityObstacle(VO._aid,VO._bid);
  DEBUG_GRADIENT("DposDvb",(VO.DposDvb()*d).norm(),(VO.DposDvb()*d-(VO2.pos()-VO.pos())/Delta).norm())
  DEBUG_GRADIENT("DnorDvb",(VO.DnorDvb()*d).norm(),(VO.DnorDvb()*d-(VO2.nor()-VO.nor())/Delta).norm())
  _perfVelocities.col(VO._bid)=old;
}
void ORCASimulator::debugDerivatives(const VelocityObstacle& VO,const Vec2T o[2]) {
  Vec2T d=Vec2T::Random(),old;
  VelocityObstacle VO2;
  DEFINE_NUMERIC_DELTA_T(T)
  //pa
  old=_agentPositions.col(VO._aid);
  _agentPositions.col(VO._aid)+=d*Delta;
  VO2=computeVelocityObstacle(VO._aid,o);
  DEBUG_GRADIENT("DposDpa",(VO.DposDpa()*d).norm(),(VO.DposDpa()*d-(VO2.pos()-VO.pos())/Delta).norm())
  DEBUG_GRADIENT("DnorDpa",(VO.DnorDpa()*d).norm(),(VO.DnorDpa()*d-(VO2.nor()-VO.nor())/Delta).norm())
  _agentPositions.col(VO._aid)=old;
  //va
  old=_perfVelocities.col(VO._aid);
  _perfVelocities.col(VO._aid)+=d*Delta;
  VO2=computeVelocityObstacle(VO._aid,o);
  DEBUG_GRADIENT("DposDva",(VO.DposDva()*d).norm(),(VO.DposDva()*d-(VO2.pos()-VO.pos())/Delta).norm())
  DEBUG_GRADIENT("DnorDva",(VO.DnorDva()*d).norm(),(VO.DnorDva()*d-(VO2.nor()-VO.nor())/Delta).norm())
  _perfVelocities.col(VO._aid)=old;
}
//helper
VelocityObstacle ORCASimulator::computeVelocityObstacle(int aid,int bid) const {
  VelocityObstacle ret;
  ret._aid=aid;
  ret._bid=bid;
  Vec2TAD pa,pb,va,vb;
  pa[0]=AD(_agentPositions(0,aid),Derivative::Unit(0));
  pa[1]=AD(_agentPositions(1,aid),Derivative::Unit(1));
  pb[0]=AD(_agentPositions(0,bid),Derivative::Unit(2));
  pb[1]=AD(_agentPositions(1,bid),Derivative::Unit(3));
  va[0]=AD(_perfVelocities(0,aid),Derivative::Unit(4));
  va[1]=AD(_perfVelocities(1,aid),Derivative::Unit(5));
  vb[0]=AD(_perfVelocities(0,bid),Derivative::Unit(6));
  vb[1]=AD(_perfVelocities(1,bid),Derivative::Unit(7));
  Vec2TAD rab=pa-pb,rabInvT=rab/_timestep,vba=vb-va;
  AD lenRab=rab.norm(),lenRabInvT=lenRab/_timestep;
  //check whether vab resides in the tip of cone
  AD cosTheta=_rad*2/lenRab;
  AD distToTip=(vba-rabInvT).norm();
  AD cosThetaTip=(vba-rabInvT).dot(-rabInvT)/distToTip/lenRabInvT;
  if(cosThetaTip.value()>cosTheta.value() || lenRab.value()<_rad*2+_gTol) {
    //in the tip
    Vec2TAD newVba=(vba-rabInvT)/distToTip*_rad*2/_timestep+rabInvT;
    ret._pos=va-(newVba-vba)/2;
    ret._nor=-(vba-rabInvT)/distToTip;
    ret._case=0;
  } else if(cross2D(rab,vba)>0) {
    //left side
    ret._nor=rot2D(AD(_rad*2),lenRab)*rab/lenRab;
    Vec2TAD newVba=vba.dot(ret._nor)*ret._nor;
    ret._pos=va-(newVba-vba)/2;
    ret._nor=Vec2TAD(ret._nor[1],-ret._nor[0]);
    ret._case=1;
  } else {
    //right side
    ret._nor=rot2D(AD(_rad*2),lenRab).transpose()*rab/lenRab;
    Vec2TAD newVba=vba.dot(ret._nor)*ret._nor;
    ret._pos=va-(newVba-vba)/2;
    ret._nor=Vec2TAD(-ret._nor[1],ret._nor[0]);
    ret._case=2;
  }
  return ret;
}
VelocityObstacle ORCASimulator::computeVelocityObstacle(int aid,const Vec2T o[2]) const {
  VelocityObstacle ret;
  ret._aid=aid;
  ret._bid=-1;
  Vec2TAD pa,va,O[2];
  pa[0]=AD(_agentPositions(0,aid),Derivative::Unit(0));
  pa[1]=AD(_agentPositions(1,aid),Derivative::Unit(1));
  va[0]=AD(_perfVelocities(0,aid),Derivative::Unit(4));
  va[1]=AD(_perfVelocities(1,aid),Derivative::Unit(5));
  for(int a=0; a<2; a++)
    for(int b=0; b<2; b++)
      O[a][b]=AD(o[a][b],Derivative::Zero());
  O[0]-=pa;
  O[1]-=pa;
  if(cross2D(O[0],O[1])>0)
    std::swap(O[0],O[1]);
  Vec2TAD nLL=rot2D(_rad,O[0].norm())*O[0];
  Vec2TAD nLR=rot2D(_rad,O[0].norm()).transpose()*O[0];
  Vec2TAD nRL=rot2D(_rad,O[1].norm())*O[1];
  Vec2TAD nRR=rot2D(_rad,O[1].norm()).transpose()*O[1];
  Vec2TAD a[2]= {Vec2TAD::Zero(),va*_timestep};
  if(cross2D(nRL,nLL)>=0 && cross2D(nRR,nLR)<=0) {
    ret._case=0;
    computeVelocityObstacle(ret,O[0],a[1],nLL,nLR);
  } else if(cross2D(nLL,nRL)>=0 && cross2D(nLR,nRR)<=0) {
    ret._case=5;
    computeVelocityObstacle(ret,O[1],a[1],nRL,nRR);
  } else {
    ret._case=10;
    computeVelocityObstacle(ret,O,a,nLL,nRR);
  }
  return ret;
}
void ORCASimulator::computeVelocityObstacle(VelocityObstacle& ret,const Vec2TAD& o,const Vec2TAD& a,const Vec2TAD& dL,const Vec2TAD& dR) const {
  Vec2TAD nL=Vec2TAD(dL[1],-dL[0]).normalized();
  Vec2TAD nR=Vec2TAD(-dR[1],dR[0]).normalized();
  AD distL=(a-o).dot(nL);
  AD distR=(a-o).dot(nR);
  if(distL>=0 && distR>=0) {
    //check 2 voronoi region
    if(distL<distR) {
      //closest feature is left border
      ret._pos=(a-nL*(distL+_rad))/_timestep;
      ret._nor=-nL;
      ret._case+=0;
    } else {
      //closest feature is right border
      ret._pos=(a-nR*(distR+_rad))/_timestep;
      ret._nor=-nR;
      ret._case+=1;
    }
  } else {
    //check 3 voronoi region
    if((a-o).dot(dL)>0) {
      //closest feature is left border
      AD distL=(a-o).dot(nL);
      ret._pos=(a-nL*(distL+_rad))/_timestep;
      ret._nor=-nL;
      ret._case+=2;
    } else if((a-o).dot(dR)>0) {
      //closest feature is right border
      AD distR=(a-o).dot(nR);
      ret._pos=(a-nR*(distR+_rad))/_timestep;
      ret._nor=-nR;
      ret._case+=3;
    } else {
      //closest feature is left vertex
      AD distVL=(a-o).norm();
      if(distVL.value()>Epsilon<T>::defaultEps())
        ret._nor=(a-o)/distVL;
      else ret._nor=(a-o)/Epsilon<T>::defaultEps();
      ret._pos=(o+ret._nor*_rad)/_timestep;
      ret._case+=4;
    }
  }
}
void ORCASimulator::computeVelocityObstacle(VelocityObstacle& ret,const Vec2TAD o[2],const Vec2TAD a[2],const Vec2TAD& dL,const Vec2TAD& dR) const {
  Vec2TAD dirO=o[1]-o[0];
  Vec2TAD nO=Vec2TAD(-dirO[1],dirO[0]).normalized();
  Vec2TAD nL=Vec2TAD(dL[1],-dL[0]).normalized();
  Vec2TAD nR=Vec2TAD(-dR[1],dR[0]).normalized();
  //find closest feature
  AD distO=(a[1]-o[0]).dot(nO);
  AD distL=(a[1]-o[0]).dot(nL);
  AD distR=(a[1]-o[1]).dot(nR);
  if(distO>=0 && distL>=0 && distR>=0) {
    //check 3 voronoi region
    if(distO<distL && distO<distR) {
      //closest feature is line segment
      ret._pos=(a[1]-nO*(distO+_rad))/_timestep;
      ret._nor=-nO;
      ret._case+=0;
    } else if(distL<distO && distL<distR) {
      //closest feature is left border
      ret._pos=(a[1]-nL*(distL+_rad))/_timestep;
      ret._nor=-nL;
      ret._case+=1;
    } else {
      //closest feature is right border
      ret._pos=(a[1]-nR*(distR+_rad))/_timestep;
      ret._nor=-nR;
      ret._case+=2;
    }
  } else {
    //check 5 voronoi region
    Vec2T oVal[2]= {Vec2T(o[0][0].value(),o[0][1].value()),Vec2T(o[1][0].value(),o[1][1].value())};
    T t=BoundingVolumeHierarchy::closestT(Vec2T(a[1][0].value(),a[1][1].value()),oVal);
    if(t<=0) {
      if((a[1]-o[0]).dot(dL)>0) {
        //closest feature is left border
        AD distL=(a[1]-o[0]).dot(nL);
        ret._pos=(a[1]-nL*(distL+_rad))/_timestep;
        ret._nor=-nL;
        ret._case+=3;
      } else {
        //closest feature is left vertex
        AD distVL=(a[1]-o[0]).norm();
        if(distVL.value()>Epsilon<T>::defaultEps())
          ret._nor=(a[1]-o[0])/distVL;
        else ret._nor=(a[1]-o[0])/Epsilon<T>::defaultEps();
        ret._pos=(o[0]+ret._nor*_rad)/_timestep;
        ret._case+=4;
      }
    } else if(t>=1) {
      if((a[1]-o[1]).dot(dR)>0) {
        //closest feature is right border
        AD distR=(a[1]-o[1]).dot(nR);
        ret._pos=(a[1]-nR*(distR+_rad))/_timestep;
        ret._nor=-nR;
        ret._case+=5;
      } else {
        //closest feature is right vertex
        AD distVR=(a[1]-o[1]).norm();
        if(distVR>Epsilon<T>::defaultEps())
          ret._nor=(a[1]-o[1])/distVR;
        else ret._nor=(a[1]-o[1])/Epsilon<T>::defaultEps();
        ret._pos=(o[1]+ret._nor*_rad)/_timestep;
        ret._case+=6;
      }
    } else {
      //closets feature is line segment
      AD distO=(a[1]-o[0]).dot(nO);
      ret._pos=(a[1]-nO*(distO+_rad))/_timestep;
      ret._nor=-nO;
      ret._case+=7;
    }
  }
}
void ORCASimulator::buildGrad(int id,MatT& DVDX,MatT& DVDV,const LPSolution& sol,const std::vector<VelocityObstacle>& VO,T tol) {
  if(sol._activeSet.first==-1) {
    //no constraints active, set identity
    DVDV.template block<2,2>(id*2,id*2).setIdentity();
  } else if(sol._activeSet.second==-1) {
    //single active constraint
    const VelocityObstacle& vo=VO[sol._activeSet.first];
    ASSERT(id==vo._aid)
    Vec2TAD v,p,n,vOut;
    v[0]=AD(sol._vIn[0],Derivative::Unit(0));
    v[1]=AD(sol._vIn[1],Derivative::Unit(1));
    p[0]=AD(vo._pos[0].value(),Derivative::Unit(2));
    p[1]=AD(vo._pos[1].value(),Derivative::Unit(3));
    n[0]=AD(vo._nor[0].value(),Derivative::Unit(4));
    n[1]=AD(vo._nor[1].value(),Derivative::Unit(5));
    Mat2T DvOutDv,DvOutDp,DvOutDn;
    //AD of single plane projection
    vOut=v-((v-p).dot(n)-tol)*n;
    //chain rule
    DvOutDv.row(0)=vOut[0].derivatives().template segment<2>(0);
    DvOutDv.row(1)=vOut[1].derivatives().template segment<2>(0);
    DvOutDp.row(0)=vOut[0].derivatives().template segment<2>(2);
    DvOutDp.row(1)=vOut[1].derivatives().template segment<2>(2);
    DvOutDn.row(0)=vOut[0].derivatives().template segment<2>(4);
    DvOutDn.row(1)=vOut[1].derivatives().template segment<2>(4);
    DVDV.template block<2,2>(id*2,id*2)+=DvOutDv;
    DVDX.template block<2,2>(id*2,vo._aid*2)+=DvOutDp*vo.DposDpa()+DvOutDn*vo.DnorDpa();
    DVDV.template block<2,2>(id*2,vo._aid*2)+=DvOutDp*vo.DposDva()+DvOutDn*vo.DnorDva();
    if(vo._bid>=0) {
      DVDX.template block<2,2>(id*2,vo._bid*2)+=DvOutDp*vo.DposDpb()+DvOutDn*vo.DnorDpb();
      DVDV.template block<2,2>(id*2,vo._bid*2)+=DvOutDp*vo.DposDvb()+DvOutDn*vo.DnorDvb();
    }
  } else {
    //two active constraints
    const VelocityObstacle& voI=VO[sol._activeSet.first];
    const VelocityObstacle& voJ=VO[sol._activeSet.second];
    ASSERT(id==voI._aid && id==voJ._aid)
    Vec2TAD v,pI,pJ,nI,nJ,vOut;
    v[0]=AD(sol._vIn[0],Derivative::Unit(0));
    v[1]=AD(sol._vIn[1],Derivative::Unit(1));
    pI[0]=AD(voI._pos[0].value(),Derivative::Unit(2));
    pI[1]=AD(voI._pos[1].value(),Derivative::Unit(3));
    nI[0]=AD(voI._nor[0].value(),Derivative::Unit(4));
    nI[1]=AD(voI._nor[1].value(),Derivative::Unit(5));
    pJ[0]=AD(voJ._pos[0].value(),Derivative::Unit(6));
    pJ[1]=AD(voJ._pos[1].value(),Derivative::Unit(7));
    nJ[0]=AD(voJ._nor[0].value(),Derivative::Unit(8));
    nJ[1]=AD(voJ._nor[1].value(),Derivative::Unit(9));
    Mat2T DvOutDv,DvOutDpI,DvOutDpJ,DvOutDnI,DvOutDnJ;
    //AD of single plane projection
    Vec2TAD RHS,lambda;
    Mat2TAD LHS=Mat2TAD::Identity();
    LHS(0,1)=LHS(1,0)=nI.dot(nJ);
    RHS[0]=nI.dot(voI.pos()-v)+tol;
    RHS[1]=nJ.dot(voJ.pos()-v)+tol;
    lambda=LHS.inverse()*RHS;
    vOut=v+nI*lambda[0]+nJ*lambda[1];
    //chain rule
    DvOutDv.row(0)=vOut[0].derivatives().template segment<2>(0);
    DvOutDv.row(1)=vOut[1].derivatives().template segment<2>(0);
    DvOutDpI.row(0)=vOut[0].derivatives().template segment<2>(2);
    DvOutDpI.row(1)=vOut[1].derivatives().template segment<2>(2);
    DvOutDnI.row(0)=vOut[0].derivatives().template segment<2>(4);
    DvOutDnI.row(1)=vOut[1].derivatives().template segment<2>(4);
    DvOutDpJ.row(0)=vOut[0].derivatives().template segment<2>(6);
    DvOutDpJ.row(1)=vOut[1].derivatives().template segment<2>(6);
    DvOutDnJ.row(0)=vOut[0].derivatives().template segment<2>(8);
    DvOutDnJ.row(1)=vOut[1].derivatives().template segment<2>(8);
    DVDV.template block<2,2>(id*2,id*2)+=DvOutDv;
    DVDX.template block<2,2>(id*2,voI._aid*2)+=DvOutDpI*voI.DposDpa()+DvOutDnI*voI.DnorDpa();
    DVDV.template block<2,2>(id*2,voI._aid*2)+=DvOutDpI*voI.DposDva()+DvOutDnI*voI.DnorDva();
    if(voI._bid>=0) {
      DVDX.template block<2,2>(id*2,voI._bid*2)+=DvOutDpI*voI.DposDpb()+DvOutDnI*voI.DnorDpb();
      DVDV.template block<2,2>(id*2,voI._bid*2)+=DvOutDpI*voI.DposDvb()+DvOutDnI*voI.DnorDvb();
    }
    DVDX.template block<2,2>(id*2,voJ._aid*2)+=DvOutDpJ*voJ.DposDpa()+DvOutDnJ*voJ.DnorDpa();
    DVDV.template block<2,2>(id*2,voJ._aid*2)+=DvOutDpJ*voJ.DposDva()+DvOutDnJ*voJ.DnorDva();
    if(voJ._bid>=0) {
      DVDX.template block<2,2>(id*2,voJ._bid*2)+=DvOutDpJ*voJ.DposDpb()+DvOutDnJ*voJ.DnorDpb();
      DVDV.template block<2,2>(id*2,voJ._bid*2)+=DvOutDpJ*voJ.DposDvb()+DvOutDnJ*voJ.DnorDvb();
    }
  }
}
bool ORCASimulator::solveActiveSet(std::pair<int,int>& activeSetInOut,Vec2T& vInOut,const std::vector<VelocityObstacle>& VO,int i,int j,T tol) {
  sort2(i,j);
  Vec2T n0=VO[i].nor();
  Vec2T n1=VO[j].nor();
  T n01=n0.dot(n1);
  if(n01>1-Epsilon<T>::defaultEps()) {
    //two planes are facing the same side
    T I=VO[i].pos().dot(VO[i].nor());
    T J=VO[j].pos().dot(VO[j].nor());
    if(I>J) {
      //plane I is closer
      activeSetInOut=std::make_pair(i,-1);
      vInOut=VO[i].proj(vInOut,tol);
      return true;
    } else if(I<J) {
      //plane J is closer
      activeSetInOut=std::make_pair(j,-1);
      vInOut=VO[j].proj(vInOut,tol);
      return true;
    } else {
      //keep active set as is
      return true;
    }
  } else if(n01<-1+Epsilon<T>::defaultEps()) {
    //inside two opposite planes, no solution
    return false;
  } else {
    //solve equations simultaneously
    Vec2T nI=VO[i].nor();
    Vec2T nJ=VO[j].nor();
    Vec2T RHS,lambda;
    Mat2T LHS=Mat2T::Identity();
    LHS(0,1)=LHS(1,0)=nI.dot(nJ);
    RHS[0]=nI.dot(VO[i].pos()-vInOut)+tol;
    RHS[1]=nJ.dot(VO[j].pos()-vInOut)+tol;
    lambda=LHS.inverse()*RHS;
    vInOut+=nI*lambda[0]+nJ*lambda[1];
    activeSetInOut=std::make_pair(i,j);
    return true;
  }
}
bool ORCASimulator::updateActiveSet(LPSolution& sol,const std::vector<VelocityObstacle>& VO,int i,T tol) {
  if(i==sol._activeSet.first || i==sol._activeSet.second) {
    //no update needed
    return false;
  } else if(sol._activeSet.first==-1 && sol._activeSet.second==-1) {
    //no active set, acquire this one
    sol._vOut=VO[i].proj(sol._vIn,tol);
    sol._activeSet.first=i;
    return true;
  } else if(sol._activeSet.second==-1) {
    //one active set, acquire this one
    Vec2T vInOut=sol._vIn;
    std::pair<int,int> activeSet=sol._activeSet;
    if(!solveActiveSet(activeSet,vInOut,VO,sol._activeSet.first,i,tol)) {
      //failed, LP has no solution
      sol._succ=false;
      return false;
    } else if(activeSet!=sol._activeSet) {
      //active set updated
      sol._activeSet=activeSet;
      sol._vOut=vInOut;
      return true;
    } else {
      //did not update solution
      return false;
    }
  } else {
    //two active set, update and check improvement
    T vio=violation(sol._vOut,VO);
    //try to replace the second index
    Vec2T vInOutA=sol._vIn;
    std::pair<int,int> activeSetA=sol._activeSet;
    if(!solveActiveSet(activeSetA,vInOutA,VO,sol._activeSet.first,i,tol)) {
      sol._succ=false;
      return false;
    } else if(violation(vInOutA,VO)<vio) {
      //this implies smaller violation
      sol._activeSet=activeSetA;
      sol._vOut=vInOutA;
      return true;
    }
    //try to replace the first index
    Vec2T vInOutB=sol._vIn;
    std::pair<int,int> activeSetB=sol._activeSet;
    if(!solveActiveSet(activeSetB,vInOutB,VO,sol._activeSet.second,i,tol)) {
      sol._succ=false;
      return false;
    } else if(violation(vInOutB,VO)<vio) {
      //this implies smaller violation
      sol._activeSet=activeSetB;
      sol._vOut=vInOutB;
      return true;
    }
    return false;
  }
}
LPSolution ORCASimulator::solveLP(const Vec2T& v,const std::vector<VelocityObstacle>& VO,T tol) {
  bool updated=true;
  LPSolution sol;
  sol._vIn=sol._vOut=v;
  sol._activeSet=std::make_pair(-1,-1);
  sol._succ=true;
  int iter=0;
  while(updated && sol._succ) {
    updated=false;
    //acquire new active set
    for(int i=0; i<(int)VO.size(); i++)
      if(!VO[i].outside(sol._vOut) && updateActiveSet(sol,VO,i,tol)) {
        updated=true;
        break;
      }
    iter++;
  }
  if(violation(sol._vOut,VO)>0)
    sol._succ=false;
  return sol;
}
ORCASimulator::T ORCASimulator::violation(const Vec2T& vOut,const std::vector<VelocityObstacle>& VO) {
  T vio=0;
  for(const auto& vo:VO)
    vio=fmax(vio,vo.violation(vOut));
  return vio;
}
ORCASimulator::T ORCASimulator::cross2D(const Vec2TAD& a,const Vec2TAD& b) {
  return (a[0]*b[1]-a[1]*b[0]).value();
}
ORCASimulator::T ORCASimulator::cross2D(const Vec2T& a,const Vec2T& b) {
  return a[0]*b[1]-a[1]*b[0];
}
ORCASimulator::Mat2TAD ORCASimulator::rot2D(AD dist,AD cord) {
  AD s=dist/cord;
  if(s.value()<=0)
    s=AD(0,Derivative::Zero());
  else if(s.value()>=1-Epsilon<T>::defaultEps())
    s=AD(1-Epsilon<T>::defaultEps(),Derivative::Zero());
  AD c=sqrt(1-s*s);
  Mat2TAD ret;
  ret(0,0)=ret(1,1)=c;
  ret(0,1)=ret(1,0)=s;
  ret(0,1)*=-1;
  return ret;
}
}

#include "ORCA.h"
#include "Epsilon.h"
#include "BoundingVolumeHierarchy.h"
#include "SpatialHashLinkedList.h"
#include "SpatialHashRadixSort.h"
#include <iostream>

namespace RVO {
ORCASimulator::ORCASimulator(const ORCASimulator& other):RVOSimulator(other) {}
ORCASimulator& ORCASimulator::operator=(const ORCASimulator& other) {
  RVOSimulator::operator=(other);
  return *this;
}
ORCASimulator::ORCASimulator(T rad,T d0,T gTol,T coef,T timestep,int maxIter,bool radixSort,bool useHash)
  :RVOSimulator(rad,d0,gTol,coef,timestep,maxIter,radixSort,useHash) {}
bool ORCASimulator::optimize(bool requireGrad,bool output) {
  //Stage 1: build hash
  //we need to build a hash spanning the largest influence range: v*dt+r
  if(output)
    std::cout << "Computing agent hash!" << std::endl;
  Vec prevPositions=Vec::Zero(_agentPositions.size());
  Vec nextPositions=Vec::Zero(_agentPositions.size());
  Eigen::Map<Mat2XT>(prevPositions.data(),2,_agentPositions.cols())=_agentPositions-_perfVelocities*_timestep;
  Eigen::Map<Mat2XT>(nextPositions.data(),2,_agentPositions.cols())=_agentPositions+_perfVelocities*_timestep;
  _hash->buildSpatialHash(mapCV(prevPositions),mapCV(nextPositions),_rad);
  //Stage 2: compute velocity obstacles between all pairs of agents
  if(output)
    std::cout << "Computing agent velocity obstacles!" << std::endl;
  auto computeVAFunc=[&](AgentNeighbor n)->bool{
    VelocityObstacle VO0=computeVelocityObstacle(n._v[0]->_id,n._v[1]->_id,requireGrad);
    VelocityObstacle VO1=computeVelocityObstacle(n._v[1]->_id,n._v[0]->_id,requireGrad);
    updateLP(VO0);
    updateLP(VO1);
    return true;
  };
  if(_useHash)
    _hash->detectSphereBroad(computeVAFunc,*_hash,0);
  else _hash->detectSphereBroadBF(computeVAFunc,*_hash,0);
  //Stage 3: compute velocity obstacles between all pairs of agent-and-obstacles
  if(output)
    std::cout << "Computing agent-obstacle velocity obstacles!" << std::endl;
  auto computeVOFunc=[&](AgentObstacleNeighbor n)->bool {
    Vec2T obs[2]={n._o->_pos,n._o->_next->_pos};
    VelocityObstacle VO=computeVelocityObstacle(n._v->_id,obs,requireGrad);
    updateLP(VO);
    return true;
  };
  if(_useHash)
    _hash->detectImplicitShape(computeVOFunc,_bvh,0);
  else _hash->detectImplicitShapeBF(computeVOFunc,_bvh,0);
  //Stage 4: adjust velocity
  if(output)
    std::cout << "Adjusting velocities!" << std::endl;
  return true;
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
    VelocityObstacle VO0=computeVelocityObstacle(aid,bid,true);
    VelocityObstacle VO1=computeVelocityObstacle(bid,aid,true);
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
    VelocityObstacle VO0=computeVelocityObstacle(aid,bid,true);
    VelocityObstacle VO1=computeVelocityObstacle(bid,aid,true);
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
    VelocityObstacle VO0=computeVelocityObstacle(aid,bid,true);
    VelocityObstacle VO1=computeVelocityObstacle(bid,aid,true);
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
    Vec2T oCopy[2]= {o[0],o[1]};
    VelocityObstacle VO=computeVelocityObstacle(aid,oCopy,true);
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
    Vec2T oCopy[2]= {o[0],o[1]};
    VelocityObstacle VO=computeVelocityObstacle(aid,oCopy,true);
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
    Vec2T oCopy[2]= {o[0],o[1]};
    VelocityObstacle VO=computeVelocityObstacle(aid,oCopy,true);
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
//helper
ORCASimulator::VelocityObstacle ORCASimulator::computeVelocityObstacle(int aid,int bid,bool requireGrad) const {
  VelocityObstacle ret;
  ret._aid=aid;
  ret._bid=bid;
  if(requireGrad) {
    ret._DvnDpva.setZero();
    ret._DvnDpvb.setZero();
  }
  Vec2T rab=_agentPositions.col(aid)-_agentPositions.col(bid),rabInvT=rab/_timestep;
  Vec2T vba=_perfVelocities.col(bid)-_perfVelocities.col(aid);
  T lenRab=rab.norm(),lenRabInvT=lenRab/_timestep;
  //check whether vab resides in the tip of cone
  T cosTheta=_rad*2/lenRab;
  T distToTip=(vba-rabInvT).norm();
  T cosThetaTip=(vba-rabInvT).dot(-rabInvT)/distToTip/lenRabInvT;
  if(cosThetaTip>cosTheta) {
    //in the tip
    Vec2T newVba=(vba-rabInvT)/distToTip*_rad*2/_timestep+rabInvT;
    ret.pos()=_perfVelocities.col(ret._aid)-(newVba-vba)/2;
    ret.nor()=-(vba-rabInvT)/distToTip;
    ret._case=0;
  } else if(cross2D(rab,vba)>0) {
    //left side
    ret.nor()=rot2D(_rad*2,lenRab)*rab/lenRab;
    Vec2T newVba=vba.dot(ret.nor())*ret.nor();
    ret.pos()=_perfVelocities.col(ret._aid)-(newVba-vba)/2;
    ret.nor()=Vec2T(ret.nor()[1],-ret.nor()[0]);
    ret._case=1;
  } else {
    //right side
    ret.nor()=rot2D(_rad*2,lenRab).transpose()*rab/lenRab;
    Vec2T newVba=vba.dot(ret.nor())*ret.nor();
    ret.pos()=_perfVelocities.col(ret._aid)-(newVba-vba)/2;
    ret.nor()=Vec2T(-ret.nor()[1],ret.nor()[0]);
    ret._case=2;
  }
  return ret;
}
ORCASimulator::VelocityObstacle ORCASimulator::computeVelocityObstacle(int aid,Vec2T o[2],bool requireGrad) const {
  VelocityObstacle ret;
  ret._aid=aid;
  ret._bid=-1;
  if(requireGrad) {
    ret._DvnDpva.setZero();
    ret._DvnDpvb.setZero();
  }
  o[0]-=_agentPositions.col(aid);
  o[1]-=_agentPositions.col(aid);
  if(cross2D(o[0],o[1])>0)
    std::swap(o[0],o[1]);
  Vec2T nLL=rot2D(_rad,o[0].norm())*o[0];
  Vec2T nLR=rot2D(_rad,o[0].norm()).transpose()*o[0];
  Vec2T nRL=rot2D(_rad,o[1].norm())*o[1];
  Vec2T nRR=rot2D(_rad,o[1].norm()).transpose()*o[1];
  if(cross2D(nRL,nLL)>=0 && cross2D(nRR,nLR)<=0) {
    ret._case=0;
    computeVelocityObstacle(ret,o[0],nLL,nLR,requireGrad);
  } else if(cross2D(nLL,nRL)>=0 && cross2D(nLR,nRR)<=0) {
    ret._case=5;
    computeVelocityObstacle(ret,o[1],nRL,nRR,requireGrad);
  } else {
    ret._case=10;
    computeVelocityObstacle(ret,o,nLL,nRR,requireGrad);
  }
  return ret;
}
void ORCASimulator::computeVelocityObstacle(VelocityObstacle& ret,const Vec2T& o,const Vec2T& dL,const Vec2T& dR,bool requireGrad) const {
  Vec2T nL=Vec2T(dL[1],-dL[0]).normalized();
  Vec2T nR=Vec2T(-dR[1],dR[0]).normalized();
  Vec2T a=_perfVelocities.col(ret._aid)*_timestep;
  T distL=(a-o).dot(nL);
  T distR=(a-o).dot(nR);
  if(distL>=0 && distR>=0) {
    //check 2 voronoi region
    if(distL<distR) {
      //closest feature is left border
      ret.pos()=(a-nL*(distL+_rad))/_timestep;
      ret.nor()=-nL;
      ret._case+=0;
    } else {
      //closest feature is right border
      ret.pos()=(a-nR*(distR+_rad))/_timestep;
      ret.nor()=-nR;
      ret._case+=1;
    }
  } else {
    //check 3 voronoi region
    if((a-o).dot(dL)>0) {
      //closest feature is left border
      T distL=(a-o).dot(nL);
      ret.pos()=(a-nL*(distL+_rad))/_timestep;
      ret.nor()=-nL;
      ret._case+=2;
    } else if((a-o).dot(dR)>0) {
      //closest feature is right border
      T distR=(a-o).dot(nR);
      ret.pos()=(a-nR*(distR+_rad))/_timestep;
      ret.nor()=-nR;
      ret._case+=3;
    } else {
      //closest feature is left vertex
      T distVL=(a-o).norm();
      ret.nor()=(a-o)/fmax(distVL,Epsilon<T>::defaultEps());
      ret.pos()=(o+ret.nor()*_rad)/_timestep;
      ret._case+=4;
    }
  }
}
void ORCASimulator::computeVelocityObstacle(VelocityObstacle& ret,const Vec2T o[2],const Vec2T& dL,const Vec2T& dR,bool requireGrad) const {
  Vec2T dirO=o[1]-o[0];
  Vec2T nO=Vec2T(-dirO[1],dirO[0]).normalized();
  Vec2T nL=Vec2T(dL[1],-dL[0]).normalized();
  Vec2T nR=Vec2T(-dR[1],dR[0]).normalized();
  //find closest feature
  Vec2T a[2]= {Vec2T::Zero(),_perfVelocities.col(ret._aid)*_timestep};
  T distO=(a[1]-o[0]).dot(nO);
  T distL=(a[1]-o[0]).dot(nL);
  T distR=(a[1]-o[1]).dot(nR);
  if(distO>=0 && distL>=0 && distR>=0) {
    //check 3 voronoi region
    if(distO<distL && distO<distR) {
      //closest feature is line segment
      ret.pos()=(a[1]-nO*(distO+_rad))/_timestep;
      ret.nor()=-nO;
      ret._case+=0;
    } else if(distL<distO && distL<distR) {
      //closest feature is left border
      ret.pos()=(a[1]-nL*(distL+_rad))/_timestep;
      ret.nor()=-nL;
      ret._case+=1;
    } else {
      //closest feature is right border
      ret.pos()=(a[1]-nR*(distR+_rad))/_timestep;
      ret.nor()=-nR;
      ret._case+=2;
    }
  } else {
    //check 5 voronoi region
    T t=BoundingVolumeHierarchy::closestT(a[1],o);
    if(t<=0) {
      if((a[1]-o[0]).dot(dL)>0) {
        //closest feature is left border
        T distL=(a[1]-o[0]).dot(nL);
        ret.pos()=(a[1]-nL*(distL+_rad))/_timestep;
        ret.nor()=-nL;
        ret._case+=3;
      } else {
        //closest feature is left vertex
        T distVL=(a[1]-o[0]).norm();
        ret.nor()=(a[1]-o[0])/fmax(distVL,Epsilon<T>::defaultEps());
        ret.pos()=(o[0]+ret.nor()*_rad)/_timestep;
        ret._case+=4;
      }
    } else if(t>=1) {
      if((a[1]-o[1]).dot(dR)>0) {
        //closest feature is right border
        T distR=(a[1]-o[1]).dot(nR);
        ret.pos()=(a[1]-nR*(distR+_rad))/_timestep;
        ret.nor()=-nR;
        ret._case+=5;
      } else {
        //closest feature is right vertex
        T distVR=(a[1]-o[1]).norm();
        ret.nor()=(a[1]-o[1])/fmax(distVR,Epsilon<T>::defaultEps());
        ret.pos()=(o[1]+ret.nor()*_rad)/_timestep;
        ret._case+=6;
      }
    } else {
      //closets feature is line segment
      T distO=(a[1]-o[0]).dot(nO);
      ret.pos()=(a[1]-nO*(distO+_rad))/_timestep;
      ret.nor()=-nO;
      ret._case+=7;
    }
  }
}
ORCASimulator::T ORCASimulator::cross2D(const Vec2T& a,const Vec2T& b) {
  return a[0]*b[1]-a[1]*b[0];
}
ORCASimulator::Mat2T ORCASimulator::rot2D(T dist,T cord) {
  T s=fmax((T)0,fmin(1-Epsilon<T>::defaultEps(),dist/cord)),c=sqrt(1-s*s);
  Mat2T ret;
  ret(0,0)=ret(1,1)=c;
  ret(0,1)=ret(1,0)=s;
  ret(0,1)*=-1;
  return ret;
}
}

#include "SourceSink.h"
#include "SpatialHash.h"

namespace RVO {
void SourceSink::addSourceSink(const Vec2T& source,const Vec2T& target,const BBox& sink,T rad) {
  _sourcePos.add(source);
  _targetPos.add(target);
  _sinkRegion.add(sink._minC);
  _sinkRegion.add(sink._maxC);
  _rad.add(rad);
  _id.add(_sourcePos.cols()-1);
}
void SourceSink::removeAgents(RVOSimulator& sim) {
  std::vector<int> delId;
  for(int i=0; i<sim.getAgentId().size();) {
    const Vec2T p=sim.getAgentPosition(i);
    const int idS=sim.getAgentId(i)%_sourcePos.cols();
    const Vec2T minC=_sinkRegion.getCMap().col(idS*2+0);
    const Vec2T maxC=_sinkRegion.getCMap().col(idS*2+1);
    if((p.array()>=minC.array()).all() && (p.array()<=maxC.array()).all())
      delId.push_back(i);
    else i++;
  }
}
void SourceSink::addAgents(RVOSimulator& sim,T eps) {
  std::vector<char> collide;
  VecCM pos=mapV2CV(sim.getAgentPositionsVec());
  sim.getHash()->buildSpatialHash(pos,pos,sim.getMaxRadius());
  sim.getHash()->detectSphereBroad(collide,_sourcePos,_rad,eps);
  for(int i=0; i<_sourcePos.cols(); i++)
    if(!collide[i]) {
      const Vec2T p=_sourcePos.getCMap().col(i);
      const Vec2T t=_targetPos.getCMap().col(i);
      const T r=_rad.getCMap()[i];
      const int id=_id.getCMap()[i];
      sim.addAgent(p,Vec2T::Zero(),r,id);
      sim.setAgentTarget(sim.getNrAgent()-1,t,_maxVelocity);
      _id.getMap()[i]+=_sourcePos.cols();
    }
}
void SourceSink::reset(T maxVelocity) {
  for(int i=0; i<_sourcePos.cols(); i++)
    _id.getMap()[i]=i;
  _maxVelocity=maxVelocity;
}
}

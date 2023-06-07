#include "SourceSink.h"
#include "SpatialHash.h"

namespace RVO {
//Trajectory
Trajectory::Trajectory():_terminated(false) {}
//SourceSink
SourceSink::SourceSink(T maxVelocity,int maxBatch):_maxVelocity(maxVelocity),_maxBatch(maxBatch) {}
std::vector<Trajectory> SourceSink::getTrajectories() const {
  return _trajectories;
}
void SourceSink::addSourceSink(const Vec2T& source,const Vec2T& target,const BBox& sink,T rad) {
  _sourcePos.add(source);
  _targetPos.add(target);
  _sinkRegion.add(sink._minC);
  _sinkRegion.add(sink._maxC);
  _rad.add(rad);
  _id.add(_sourcePos.cols()-1);
}
void SourceSink::addAgents(RVOSimulator& sim,T eps) {
  std::vector<char> collide;
  VecCM pos=mapV2CV(sim.getAgentPositionsVec());
  if(sim.getNrAgent()>0) {
    sim.getHash()->buildSpatialHash(pos,pos,sim.getMaxRadius());
    sim.getHash()->detectSphereBroad(collide,_sourcePos,_rad,eps);
  } else collide.assign(_sourcePos.cols(),false);
  for(int i=0; i<_sourcePos.cols(); i++)
    if(!collide[i]) {
      const Vec2T p=_sourcePos.getCMap().col(i);
      const Vec2T t=_targetPos.getCMap().col(i);
      const T r=_rad.getCMap()[i];
      const int id=_id.getCMap()[i];
      if(id>=_maxBatch*_sourcePos.cols())
        continue;
      sim.addAgent(p,Vec2T::Zero(),r,id);
      sim.setAgentTarget(sim.getNrAgent()-1,t,_maxVelocity);
      _id.getMap()[i]+=_sourcePos.cols();
      //record: initialize trajectory
      if((int)_trajectories.size()<=id)
        _trajectories.resize(id+1,Trajectory());
      _trajectories[id]._terminated=false;
      _trajectories[id]._pos.push_back(p);
      _trajectories[id]._target=t;
      _trajectories[id]._rad=r;
    }
}
void SourceSink::recordAgents(const RVOSimulator& sim) {
  for(int i=0; i<sim.getNrAgent(); i++) {
    const Vec2T p=sim.getAgentPosition(i);
    const int id=sim.getAgentId(i);
    _trajectories[id]._pos.push_back(p);
  }
}
void SourceSink::removeAgents(RVOSimulator& sim) {
  for(int i=0; i<sim.getAgentId().size();) {
    const Vec2T p=sim.getAgentPosition(i);
    const int id=sim.getAgentId(i);
    const int idS=id%_sourcePos.cols();
    const Vec2T minC=_sinkRegion.getCMap().col(idS*2+0);
    const Vec2T maxC=_sinkRegion.getCMap().col(idS*2+1);
    if((p.array()>=minC.array()).all() && (p.array()<=maxC.array()).all()) {
      _trajectories[id]._terminated=true;
      sim.removeAgent(i);
    } else i++;
  }
}
void SourceSink::reset() {
  for(int i=0; i<_sourcePos.cols(); i++)
    _id.getMap()[i]=i;
  _trajectories.clear();
}
}

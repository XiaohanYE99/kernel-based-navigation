#include "SourceSink.h"
#include "SpatialHash.h"

namespace RVO {
//Trajectory
Trajectory::Trajectory():_terminated(false) {}
int Trajectory::startFrame() const {
  return _startFrame;
}
int Trajectory::endFrame() const {
  return _startFrame+(int)_pos.size();
}
bool Trajectory::terminated() const {
  return _terminated;
}
Trajectory::Mat2XT Trajectory::pos() const {
  Mat2XT ret;
  ret.resize(2,(int)_pos.size());
  for(int i=0; i<(int)_pos.size(); i++)
    ret.col(i)=_pos[i];
  return ret;
}
Trajectory::Vec2T Trajectory::target() const {
  return _target;
}
Trajectory::T Trajectory::rad() const {
  return _rad;
}
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
std::pair<SourceSink::Mat2XT,SourceSink::Vec> SourceSink::getAgentPositions(int frameId,const std::vector<Trajectory>& trajectories) {
  //count particle
  int n=0;
  for(const auto& t:trajectories)
    if(t._startFrame<=frameId && frameId<t.endFrame())
      n++;
  //fill data
  std::pair<Mat2XT,Vec> ret;
  ret.first.resize(2,n);
  ret.second.resize(n);
  n=0;
  for(const auto& t:trajectories)
    if(t._startFrame<=frameId && frameId<t.endFrame()) {
      ret.first.col(n)=t._pos[frameId-t._startFrame];
      ret.second[n]=t._rad;
      n++;
    }
  return ret;
}
void SourceSink::addAgents(int frameId,RVOSimulator& sim,T eps) {
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
      int aid=sim.addAgent(p,Vec2T::Zero(),r,id);
      sim.setAgentTarget(aid,t,_maxVelocity);
      _id.getMap()[i]+=_sourcePos.cols();
      //record: initialize trajectory
      if((int)_trajectories.size()<=id)
        _trajectories.resize(id+1,Trajectory());
      _trajectories[id]._startFrame=frameId;
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
    if(id>=0 && id<(int)_trajectories.size())
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

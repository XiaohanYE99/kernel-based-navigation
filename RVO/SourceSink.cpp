#include "SourceSink.h"
#include "SpatialHash.h"

namespace RVO {
//Trajectory
Trajectory::Trajectory():_endFrame(-1),_startFrame(-1),_terminated(false),_recordFull(true) {}
Trajectory::Trajectory(bool recordFull,int frameId,const Vec2T& target,T r)
  :_endFrame(frameId),_startFrame(frameId),_terminated(false),_recordFull(recordFull),_target(target),_rad(r) {}
int Trajectory::startFrame() const {
  return _startFrame;
}
int Trajectory::endFrame() const {
  return _endFrame;
}
bool Trajectory::isFullTrajectory() const {
  return _recordFull;
}
bool Trajectory::terminated() const {
  return _terminated;
}
void Trajectory::terminate() {
  _terminated=true;
}
void Trajectory::addPos(const Vec2T& pos) {
  if(_recordFull) {
    _fullPos.push_back(pos);
    _endFrame++;
  } else {
    if(_endFrame==_startFrame)
      _startEndPos.col(0)=pos;
    _startEndPos.col(1)=pos;
    _endFrame++;
  }
}
Trajectory::Vec2T Trajectory::pos(int frameId) const {
  if(_recordFull) {
    ASSERT_MSGV(frameId>=_startFrame && frameId<_endFrame,
                "FullTrajectory frameId(%d) out of the range of [%d,%d)!",frameId,_startFrame,_endFrame)
    return _fullPos[frameId-_startFrame];
  } else {
    ASSERT_MSGV(frameId==_startFrame || frameId==_endFrame-1,
                "StartEndTrajectory frameId(%d) out of the range of %d/%d-1!",frameId,_startFrame,_endFrame)
    if(frameId==_startFrame)
      return _startEndPos.col(0);
    else return _startEndPos.col(1);
  }
}
Trajectory::Mat2XT Trajectory::pos() const {
  if(_recordFull) {
    Mat2XT ret;
    ret.resize(2,(int)_fullPos.size());
    for(int i=0; i<(int)_fullPos.size(); i++)
      ret.col(i)=_fullPos[i];
    return ret;
  } else return _startEndPos;
}
Trajectory::Vec2T Trajectory::target() const {
  return _target;
}
Trajectory::T Trajectory::rad() const {
  return _rad;
}
//SourceSink
SourceSink::SourceSink(T maxVelocity,int maxBatch,bool recordFull):_maxVelocity(maxVelocity),_maxBatch(maxBatch),_recordFull(recordFull) {}
const DynamicMat<SourceSink::T>& SourceSink::getSourcePos() const {
  return _sourcePos;
}
const DynamicMat<SourceSink::T>& SourceSink::getTargetPos() const {
  return _targetPos;
}
const DynamicMat<SourceSink::T>& SourceSink::getSinkRegion() const {
  return _sinkRegion;
}
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
    if(t.startFrame()<=frameId && frameId<t.endFrame())
      n++;
  //fill data
  std::pair<Mat2XT,Vec> ret;
  ret.first.resize(2,n);
  ret.second.resize(n);
  n=0;
  for(const auto& t:trajectories)
    if(t.startFrame()<=frameId && frameId<t.endFrame()) {
      ret.first.col(n)=t.pos(frameId);
      ret.second[n]=t.rad();
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
      _trajectories[id]=Trajectory(_recordFull,frameId,t,r);
      _trajectories[id].addPos(p);
    }
}
void SourceSink::recordAgents(const RVOSimulator& sim) {
  for(int i=0; i<sim.getNrAgent(); i++) {
    const Vec2T p=sim.getAgentPosition(i);
    const int id=sim.getAgentId(i);
    if(id>=0 && id<(int)_trajectories.size())
      _trajectories[id].addPos(p);
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
      _trajectories[id].terminate();
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

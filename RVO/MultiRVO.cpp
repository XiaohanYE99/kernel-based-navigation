#include "MultiRVO.h"
#include "Scan.h"
#define ASSERT_SOURCE_SINK ASSERT_MSGV(!_sss.empty(),"When you have not setup SourceSink, you cannot call %s!",__FUNCTION__)
#define ASSERT_NO_SOURCE_SINK ASSERT_MSGV(_sss.empty(),"When you have setup SourceSink, you cannot call %s because each environment has different number of agents!",__FUNCTION__)

namespace RVO {
MultiRVOSimulator::MultiRVOSimulator(int batchSize,T d0,T gTol,T coef,T timestep,int maxIter,bool radixSort,bool useHash,const std::string& optimizer) {
  for(int i=0; i<batchSize; i++)
    _sims.push_back(RVOSimulator(d0,gTol,coef,timestep,maxIter,radixSort,useHash,optimizer));
  clearAgent();
}
void MultiRVOSimulator::clearAgent() {
  for(int i=0; i<(int)_sims.size(); i++) {
    _sims[i].clearAgent();
    if(!_sss.empty())
      _sss[i].reset();
  }
  _frameId=0;
}
void MultiRVOSimulator::clearObstacle() {
  for(auto& sim:_sims)
    sim.clearObstacle();
}
int MultiRVOSimulator::getNrObstacle() const {
  return _sims[0].getNrObstacle();
}
std::vector<MultiRVOSimulator::Vec2T> MultiRVOSimulator::getObstacle(int i) const {
  return _sims[0].getObstacle(i);
}
int MultiRVOSimulator::getNrAgent() const {
  return _sims[0].getNrAgent();
}
void MultiRVOSimulator::setupSourceSink(T maxVelocity,int maxBatch) {
  _sss.assign(_sims.size(),SourceSink(maxVelocity,maxBatch));
}
std::vector<Trajectory> MultiRVOSimulator::getTrajectories(int id) const {
  ASSERT_SOURCE_SINK
  return _sss[id].getTrajectories();
}
std::vector<std::vector<Trajectory>> MultiRVOSimulator::getAllTrajectories() const {
  ASSERT_SOURCE_SINK
  std::vector<std::vector<Trajectory>> trajs(_sss.size());
  for(int i=0; i<(int)_sss.size(); i++)
    trajs[i]=_sss[i].getTrajectories();
  return trajs;
}
void MultiRVOSimulator::addSourceSink(Vec2T source,Vec2T target,Vec2T minC,Vec2T maxC,T rad,T noise) {
  ASSERT_SOURCE_SINK
  for(int i=0; i<(int)_sss.size(); i++)
    _sss[i].addSourceSink(source+Vec2T::Random()*noise,target,BBox(minC,maxC),rad);
}
void MultiRVOSimulator::setAllAgentVelocities(int id,Mat2XT vel) {
  for(int c=0; c<vel.cols(); c++)
    _sims[id].setAgentVelocity(c,vel.col(c));
}
void MultiRVOSimulator::setAllAgentBatchVelocities(Mat2XT vel) {
  _nrA.assign(_sims.size()+1,0);
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_sims.size(); i++)
    _nrA[i+1]=_sims[i].getNrAgent();
  //scan to get offset
  omp_scan_add(_nrA,_offA);
  //set velocities
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_sims.size(); i++)
    _sims[i].getAgentVelocities()=vel.block(0,_offA[i],2,_nrA[i+1]);
}
MultiRVOSimulator::Mat2XT MultiRVOSimulator::getAllAgentPositions(int id) const {
  return _sims[id].getAgentPositions();
}
MultiRVOSimulator::Mat2XT MultiRVOSimulator::getAllAgentBatchPositions() {
  _nrA.assign(_sims.size()+1,0);
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_sims.size(); i++)
    _nrA[i+1]=_sims[i].getNrAgent();
  //scan to get offset
  omp_scan_add(_nrA,_offA);
  //set positions
  Mat2XT pos;
  pos.resize(2,_offA.back());
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_sims.size(); i++)
    pos.block(0,_offA[i],2,_nrA[i+1])=_sims[i].getAgentPositions();
  return pos;
}
MultiRVOSimulator::Mat2XT MultiRVOSimulator::getAllAgentTargets(int id) const {
  return _sims[id].getAgentTargets();
}
MultiRVOSimulator::Mat2XT MultiRVOSimulator::getAllAgentBatchTargets() {
  _nrA.assign(_sims.size()+1,0);
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_sims.size(); i++)
    _nrA[i+1]=_sims[i].getNrAgent();
  //scan to get offset
  omp_scan_add(_nrA,_offA);
  //set positions
  Mat2XT tar;
  tar.resize(2,_offA.back());
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_sims.size(); i++)
    tar.block(0,_offA[i],2,_nrA[i+1])=_sims[i].getAgentTargets();
  return tar;
}
MultiRVOSimulator::Veci MultiRVOSimulator::getAllAgentBatchIds() {
  _nrA.assign(_sims.size()+1,0);
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_sims.size(); i++)
    _nrA[i+1]=_sims[i].getNrAgent();
  //scan to get offset
  omp_scan_add(_nrA,_offA);
  //set ids
  Veci ids;
  ids.resize(_offA.back());
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)_sims.size(); i++)
    ids.segment(_offA[i],_nrA[i+1]).setConstant(i);
  return ids;
}
std::vector<MultiRVOSimulator::Vec2T> MultiRVOSimulator::getAgentPosition(int i) const {
  ASSERT_NO_SOURCE_SINK
  std::vector<Vec2T> pos;
  for(auto& sim:_sims)
    pos.push_back(sim.getAgentPosition(i));
  return pos;
}
std::vector<MultiRVOSimulator::Vec2T> MultiRVOSimulator::getAgentVelocity(int i) const {
  ASSERT_NO_SOURCE_SINK
  std::vector<Vec2T> vel;
  for(auto& sim:_sims)
    vel.push_back(sim.getAgentVelocity(i));
  return vel;
}
std::vector<MultiRVOSimulator::Mat2T> MultiRVOSimulator::getAgentDVDP(int i) const {
  ASSERT_NO_SOURCE_SINK
  std::vector<Mat2T> DVDP;
  for(auto& sim:_sims)
    DVDP.push_back(sim.getAgentDVDP(i));
  return DVDP;
}
std::vector<MultiRVOSimulator::T> MultiRVOSimulator::getAgentRadius(int i) const {
  ASSERT_NO_SOURCE_SINK
  std::vector<T> rad;
  for(auto& sim:_sims)
    rad.push_back(sim.getAgentRadius(i));
  return rad;
}
int MultiRVOSimulator::addAgent(std::vector<Vec2T> pos,std::vector<Vec2T> vel,std::vector<T> rad) {
  ASSERT_NO_SOURCE_SINK
  for(int i=0; i<(int)_sims.size(); i++)
    _sims[i].addAgent(pos[i],vel[i],rad[i]);
  return _sims[0].getNrAgent()-1;
}
void MultiRVOSimulator::setAgentPosition(int i,std::vector<Vec2T> pos) {
  ASSERT_NO_SOURCE_SINK
  for(int id=0; id<(int)_sims.size(); id++)
    _sims[id].setAgentPosition(i,pos[id]);
}
void MultiRVOSimulator::setAgentVelocity(int i,std::vector<Vec2T> vel) {
  ASSERT_NO_SOURCE_SINK
  for(int id=0; id<(int)_sims.size(); id++)
    _sims[id].setAgentVelocity(i,vel[id]);
}
void MultiRVOSimulator::setAgentTarget(int i,std::vector<Vec2T> target,T maxVelocity) {
  ASSERT_NO_SOURCE_SINK
  for(int id=0; id<(int)_sims.size(); id++)
    _sims[id].setAgentTarget(i,target[id],maxVelocity);
}
int MultiRVOSimulator::addObstacle(std::vector<Vec2T> vss) {
  int ret=-1;
  for(auto& sim:_sims)
    ret=sim.addObstacle(vss);
  return ret;
}
void MultiRVOSimulator::buildVisibility() {
  for(int i=0; i<(int)_sims.size(); i++)
    if(i==0)
      _sims[i].buildVisibility();
    else _sims[i].buildVisibility(_sims[0]);
}
void MultiRVOSimulator::clearVisibility() {
  for(int i=0; i<(int)_sims.size(); i++)
    _sims[i].clearVisibility();
}
void MultiRVOSimulator::setNewtonParameter(int maxIter,T gTol,T d0,T coef) {
  for(auto& sim:_sims)
    sim.setNewtonParameter(maxIter,gTol,d0,coef);
}
void MultiRVOSimulator::setTimestep(T timestep) {
  for(auto& sim:_sims)
    sim.setTimestep(timestep);
}
MultiRVOSimulator::T MultiRVOSimulator::timestep() const {
  return _sims[0].timestep();
}
int MultiRVOSimulator::getBatchSize() const {
  return (int)_sims.size();
}
RVOSimulator& MultiRVOSimulator::getSubSimulator(int id) {
  return _sims[id];
}
const RVOSimulator& MultiRVOSimulator::getSubSimulator(int id) const {
  return _sims[id];
}
std::vector<char> MultiRVOSimulator::optimize(bool requireGrad,bool output) {
  std::vector<char> succ(_sims.size());
  OMP_PARALLEL_FOR_
  for(int id=0; id<(int)_sims.size(); id++) {
    succ[id]=_sims[id].optimize(requireGrad,output);
    if(_sss.size()==_sims.size()) {
      _sss[id].recordAgents(_sims[id]);
      _sss[id].addAgents(_frameId,_sims[id]);
      _sss[id].removeAgents(_sims[id]);
    }
  }
  _frameId++;
  return succ;
}
void MultiRVOSimulator::updateAgentTargets() {
  OMP_PARALLEL_FOR_
  for(int id=0; id<(int)_sims.size(); id++)
    _sims[id].updateAgentTargets();
}
void MultiRVOSimulator::reset() {
  _sims.clear();
  _sss.clear();
  _frameId=0;
}
std::vector<MultiRVOSimulator::MatT> MultiRVOSimulator::getDXDX() const {
  std::vector<MatT> ret;
  for(auto& sim:_sims)
    ret.push_back(sim.getDXDX());
  return ret;
}
std::vector<MultiRVOSimulator::MatT> MultiRVOSimulator::getDXDV() const {
  std::vector<MatT> ret;
  for(auto& sim:_sims)
    ret.push_back(sim.getDXDV());
  return ret;
}
}

#include "MultiRVO.h"

namespace RVO {
MultiRVOSimulator::MultiRVOSimulator(int batchSize,T d0,T gTol,T coef,T timestep,int maxIter,bool radixSort,bool useHash,const std::string& optimizer) {
  for(int i=0; i<batchSize; i++)
    _sims.push_back(RVOSimulator(d0,gTol,coef,timestep,maxIter,radixSort,useHash,optimizer));
}
void MultiRVOSimulator::clearAgent() {
  for(auto& sim:_sims)
    sim.clearAgent();
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
std::vector<MultiRVOSimulator::Vec2T> MultiRVOSimulator::getAgentPosition(int i) const {
  std::vector<Vec2T> pos;
  for(auto& sim:_sims)
    pos.push_back(sim.getAgentPosition(i));
  return pos;
}
std::vector<MultiRVOSimulator::Vec2T> MultiRVOSimulator::getAgentVelocity(int i) const {
  std::vector<Vec2T> vel;
  for(auto& sim:_sims)
    vel.push_back(sim.getAgentVelocity(i));
  return vel;
}
std::vector<MultiRVOSimulator::T> MultiRVOSimulator::getAgentRadius(int i) const {
  std::vector<T> rad;
  for(auto& sim:_sims)
    rad.push_back(sim.getAgentRadius(i));
  return rad;
}
int MultiRVOSimulator::addAgent(std::vector<Vec2T> pos,std::vector<Vec2T> vel,std::vector<T> rad) {
  for(int i=0; i<(int)_sims.size(); i++)
    _sims[i].addAgent(pos[i],vel[i],rad[i]);
  return _sims[0].getNrAgent()-1;
}
void MultiRVOSimulator::setAgentPosition(int i,std::vector<Vec2T> pos) {
  for(int id=0; id<(int)_sims.size(); id++)
    _sims[id].setAgentPosition(i,pos[id]);
}
void MultiRVOSimulator::setAgentVelocity(int i,std::vector<Vec2T> vel) {
  for(int id=0; id<(int)_sims.size(); id++)
    _sims[id].setAgentVelocity(i,vel[id]);
}
void MultiRVOSimulator::setAgentTarget(int i,std::vector<Vec2T> target,T maxVelocity) {
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
const RVOSimulator& MultiRVOSimulator::getSubSimulator(int id) const {
  return _sims[id];
}
std::vector<char> MultiRVOSimulator::optimize(bool requireGrad,bool output) {
  std::vector<char> succ(_sims.size());
  OMP_PARALLEL_FOR_
  for(int id=0; id<(int)_sims.size(); id++)
    succ[id]=_sims[id].optimize(requireGrad,output);
  return succ;
}
void MultiRVOSimulator::updateAgentTargets() {
  OMP_PARALLEL_FOR_
  for(int id=0; id<(int)_sims.size(); id++)
    _sims[id].updateAgentTargets();
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

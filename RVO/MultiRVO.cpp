#include "MultiRVO.h"

namespace RVO {
MultiRVOSimulator::MultiRVOSimulator(int batchSize,T rad,T d0,T gTol,T coef,T timestep,int maxIter,bool radixSort,bool useHash) {
  _sims.assign(batchSize,RVOSimulator(rad,d0,gTol,coef,timestep,maxIter,radixSort,useHash));
}
MultiRVOSimulator::T MultiRVOSimulator::getRadius() const {
  return _sims[0].getRadius();
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
int MultiRVOSimulator::addAgent(const std::vector<Vec2T>& pos,const std::vector<Vec2T>& vel) {
  for(int i=0; i<(int)_sims.size(); i++)
    _sims[i].addAgent(pos[i],vel[i]);
}
void MultiRVOSimulator::setAgentPosition(int i,const std::vector<Vec2T>& pos) {
  for(int id=0; id<(int)_sims.size(); id++)
    _sims[id].setAgentPosition(i,pos[id]);
}
void MultiRVOSimulator::setAgentVelocity(int i,const std::vector<Vec2T>& vel) {
  for(int id=0; id<(int)_sims.size(); id++)
    _sims[id].setAgentVelocity(i,vel[id]);
}
void MultiRVOSimulator::setAgentTarget(int i,const std::vector<Vec2T>& target,T maxVelocity) {
  for(int id=0; id<(int)_sims.size(); id++)
    _sims[id].setAgentTarget(i,target[id],maxVelocity);
}
void MultiRVOSimulator::addObstacle(const std::vector<Vec2T>& vss) {
  for(auto& sim:_sims)
    sim.addObstacle(vss);
}
void MultiRVOSimulator::setNewtonParameter(int maxIter,T gTol,T d0,T coef) {
  for(auto& sim:_sims)
    sim.setNewtonParameter(maxIter,gTol,d0,coef);
}
void MultiRVOSimulator::setAgentRadius(T radius) {
  for(auto& sim:_sims)
    sim.setAgentRadius(radius);
}
void MultiRVOSimulator::setTimestep(T timestep) {
  for(auto& sim:_sims)
    sim.setTimestep(timestep);
}
MultiRVOSimulator::T MultiRVOSimulator::timestep() const {
  return _sims[0].timestep();
}
std::vector<char> MultiRVOSimulator::optimize(std::vector<MatT>* DXDV,std::vector<MatT>* DXDX,bool output) {
  std::vector<char> succ;
  if(DXDV)
    DXDV->resize(_sims.size());
  if(DXDX)
    DXDX->resize(_sims.size());
  for(int id=0; id<(int)_sims.size(); id++)
    succ.push_back(_sims[id].optimize(DXDV?&(*DXDV)[id]:NULL,DXDX?&(*DXDX)[id]:NULL,output));
  return succ;
}
}

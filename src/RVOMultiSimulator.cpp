/*
 * RVOMultiSimulator.cpp
 * RVO2 Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/RVO2/>
 */

#include "RVOMultiSimulator.h"
#include <iostream>
#include <omp.h>

namespace RVO {
RVOMultiSimulator::RVOMultiSimulator(size_t batch) {
  _sims.resize(batch);
  for(size_t i=0; i<_sims.size(); i++)
    _sims[i]=new RVOSimulator();
}
RVOMultiSimulator::RVOMultiSimulator(size_t batch,double timeStep, double neighborDist, size_t maxNeighbors,
                                     double timeHorizon, double timeHorizonObst, double radius,
                                     double maxSpeed, const Vector2& velocity) {
  _sims.resize(batch);
  _gradVs.resize(batch);
  _gradXs.resize(batch);
  _positions.resize(batch);
  _perfVelocities.resize(batch);
  _agentVelocities.resize(batch);
  _agentNewVelocities.resize(batch);
  for(size_t i=0; i<_sims.size(); i++)
    _sims[i]=new RVOSimulator(timeStep,neighborDist,maxNeighbors,timeHorizon,timeHorizonObst,radius,maxSpeed,velocity);
}
RVOMultiSimulator::~RVOMultiSimulator() {
  while(!_sims.empty()) {
    delete _sims.back();
    _sims.pop_back();
  }
}
size_t RVOMultiSimulator::addAgent(const std::vector<Vector2>& position) {
  size_t ret=-1;
  assert(position.size()==_sims.size());
  for(size_t i=0; i<_sims.size(); i++)
    ret=_sims[i]->addAgent(position[i]);
  return ret;
}
size_t RVOMultiSimulator::addAgentCar(const std::vector<Vector2>& position) {
  size_t ret=-1;
  assert(position.size()==_sims.size());
  for(size_t i=0; i<_sims.size(); i++)
    ret=_sims[i]->addAgentCar(position[i]);
  return ret;
}
void RVOMultiSimulator::setCarLookupTable() {
  _sims[0]->setCarLookupTable();
}
size_t RVOMultiSimulator::addAgent(const std::vector<Vector2>& position, double neighborDist,
                                   size_t maxNeighbors, double timeHorizon,
                                   double timeHorizonObst, double radius, double maxSpeed,
                                   const Vector2& velocity) {
  size_t ret=-1;
  assert(position.size()==_sims.size());
  for(size_t i=0; i<_sims.size(); i++)
    ret=_sims[i]->addAgent(position[i],neighborDist,maxNeighbors,timeHorizon,timeHorizonObst,radius,maxSpeed,velocity);
  return ret;
}
size_t RVOMultiSimulator::addAgentCar(const std::vector<Vector2>& position, double neighborDist,
                                      size_t maxNeighbors, double timeHorizon,
                                      double timeHorizonObst, double radius, double maxSpeed,
                                      const Vector2& velocity, double lambda) {
  size_t ret=-1;
  assert(position.size()==_sims.size());
  for(size_t i=0; i<_sims.size(); i++)
    ret=_sims[i]->addAgentCar(position[i],neighborDist,maxNeighbors,timeHorizon,timeHorizonObst,radius,maxSpeed,velocity,lambda);
  return ret;
}
size_t RVOMultiSimulator::addObstacle(const std::vector<Vector2>& vertices) {
  size_t ret=-1;
  for(auto& sim:_sims)
    ret=sim->addObstacle(vertices);
  return ret;
}
void RVOMultiSimulator::clearObstacle() {
  #pragma omp parallel for
  for(auto& sim:_sims)
    sim->clearObstacle();
}
void RVOMultiSimulator::doStep() {
  #pragma omp parallel for
  for(auto& sim:_sims)
    sim->doStep();
}
void RVOMultiSimulator::computeAgents() {
  #pragma omp parallel for
  for(auto& sim:_sims)
    sim->computeAgents();
}
void RVOMultiSimulator::updateAgents() {
  #pragma omp parallel for
  for(auto& sim:_sims)
    sim->updateAgents();
}
void RVOMultiSimulator::doStepCar() {
  #pragma omp parallel for
  for(auto& sim:_sims)
    sim->doStepCar();
}
void RVOMultiSimulator::setNewtonParameters(size_t maxIter, double tol, double d0, double coef, double alphaMin) {
  #pragma omp parallel for
  for(auto& sim:_sims)
    sim->setNewtonParameters(maxIter,tol,d0,coef,alphaMin);
}
void RVOMultiSimulator::doNewtonStep(bool require_grad) {
  #pragma omp parallel for
  for(int i=0;i<100;i++)
    std::cout<<i <<std::endl;
  for(auto& sim:_sims)
    sim->doNewtonStep(require_grad);
}
const std::vector<Eigen::MatrixXd>& RVOMultiSimulator::getGradV() {
  for(size_t i=0; i<_sims.size(); i++)
    _gradVs[i]=_sims[i]->getGradV();
  return _gradVs;
}
const std::vector<Eigen::MatrixXd>& RVOMultiSimulator::getGradX() {
  for(size_t i=0; i<_sims.size(); i++)
    _gradXs[i]=_sims[i]->getGradX();
  return _gradXs;
}
size_t RVOMultiSimulator::getAgentMaxNeighbors(size_t agentNo) const {
  return _sims[0]->getAgentMaxNeighbors(agentNo);
}
double RVOMultiSimulator::getAgentCarTheta(size_t agentNo) const {
  return _sims[0]->getAgentCarTheta(agentNo);
}
double RVOMultiSimulator::getAgentCarPhi(size_t agentNo) const {
  return _sims[0]->getAgentCarPhi(agentNo);
}
double RVOMultiSimulator::getAgentMaxSpeed(size_t agentNo) const {
  return _sims[0]->getAgentMaxSpeed(agentNo);
}
const std::vector<Vector2>& RVOMultiSimulator::getAgentPosition(size_t agentNo) {
  for(size_t i=0; i<_sims.size(); i++)
    _positions[i]=_sims[i]->getAgentPosition(agentNo);
  return _positions;
}
const std::vector<Vector2>& RVOMultiSimulator::getAgentPrefVelocity(size_t agentNo) {
  for(size_t i=0; i<_sims.size(); i++)
    _perfVelocities[i]=_sims[i]->getAgentPrefVelocity(agentNo);
  return _perfVelocities;
}
double RVOMultiSimulator::getAgentRadius(size_t agentNo) const {
  return _sims[0]->getAgentRadius(agentNo);
}
const std::vector<Vector2>& RVOMultiSimulator::getAgentVelocity(size_t agentNo) {
  for(size_t i=0; i<_sims.size(); i++)
    _agentVelocities[i]=_sims[i]->getAgentVelocity(agentNo);
  return _agentVelocities;
}
const std::vector<Vector2>& RVOMultiSimulator::getAgentNewVelocity(size_t agentNo) {
  for(size_t i=0; i<_sims.size(); i++)
    _agentNewVelocities[i]=_sims[i]->getAgentNewVelocity(agentNo);
  return _agentNewVelocities;
}
double RVOMultiSimulator::getGlobalTime() const {
  return _sims[0]->getGlobalTime();
}
size_t RVOMultiSimulator::getNumAgents() const {
  return _sims[0]->getNumAgents();
}
size_t RVOMultiSimulator::getNumAgentCars() const {
  return _sims[0]->getNumAgentCars();
}
size_t RVOMultiSimulator::getNumObstacleVertices() const {
  return _sims[0]->getNumObstacleVertices();
}
double RVOMultiSimulator::getTimeStep() const {
  return _sims[0]->getTimeStep();
}
void RVOMultiSimulator::processObstacles() {
  #pragma omp parallel for
  for(size_t i=0; i<_sims.size(); i++)
    _sims[i]->processObstacles();
}
bool RVOMultiSimulator::queryVisibility(const Vector2& point1, const Vector2& point2, double radius) const {
  return _sims[0]->queryVisibility(point1,point2,radius);
}
void RVOMultiSimulator::setAgentDefaults
(double neighborDist, size_t maxNeighbors,
 double timeHorizon, double timeHorizonObst,
 double radius, double maxSpeed, const Vector2& velocity) {
  for(auto& sim:_sims)
    sim->setAgentDefaults(neighborDist,maxNeighbors,timeHorizon,timeHorizonObst,radius,maxSpeed,velocity);
}
void RVOMultiSimulator::setAgentLambda(size_t agentNo, double lambda) {
  for(auto& sim:_sims)
    sim->setAgentLambda(agentNo, lambda);
}
void RVOMultiSimulator::setAgentMaxNeighbors(size_t agentNo, size_t maxNeighbors) {
  for(auto& sim:_sims)
    sim->setAgentMaxNeighbors(agentNo,maxNeighbors);
}
void RVOMultiSimulator::setAgentMaxSpeed(size_t agentNo, double maxSpeed) {
  for(auto& sim:_sims)
    sim->setAgentMaxSpeed(agentNo,maxSpeed);
}
void RVOMultiSimulator::setAgentPosition(size_t agentNo, const std::vector<Vector2>& position) {
  assert(position.size()==_sims.size());
  for(size_t i=0; i<_sims.size(); i++)
    _sims[i]->setAgentPosition(agentNo,position[i]);
}
void RVOMultiSimulator::setAgentPrefVelocity(size_t agentNo, const std::vector<Vector2>& prefVelocity) {
  for(size_t i=0; i<_sims.size(); i++)
    _sims[i]->setAgentPrefVelocity(agentNo,prefVelocity[i]);
}
void RVOMultiSimulator::setAgentRadius(size_t agentNo, double radius) {
  for(auto& sim:_sims)
    sim->setAgentRadius(agentNo,radius);
}
void RVOMultiSimulator::setAgentVelocity(size_t agentNo, const std::vector<Vector2>& velocity) {
  for(size_t i=0; i<_sims.size(); i++)
    _sims[i]->setAgentVelocity(agentNo,velocity[i]);
}
void RVOMultiSimulator::setCarProperties(double length, double radius, double vDrivingMax, double vSteeringMax, double aDrivingMax, double phiMax, double dtc, double errorPreferred, double ka, double kv, double kp, double deltaV, double deltaPhi) {
  for(size_t i=0; i<_sims.size(); i++)
    _sims[i]->setCarProperties(length,radius,vDrivingMax,vSteeringMax,aDrivingMax,phiMax,dtc,errorPreferred,ka,kv,kp,deltaV,deltaPhi);
}
void RVOMultiSimulator::setTimeStep(double timeStep) {
  for(size_t i=0; i<_sims.size(); i++)
    _sims[i]->setTimeStep(timeStep);
}
void RVOMultiSimulator::setAgentCar(size_t agentCarNo, const std::vector<Vector2>& position) {
  assert(position.size()==_sims.size());
  for(size_t i=0; i<_sims.size(); i++)
    _sims[i]->setAgentCar(agentCarNo,position[i]);
}
}

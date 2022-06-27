/*
 * RVOMultiSimulator.h
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

#ifndef RVO_MULTI_RVO_SIMULATOR_H_
#define RVO_MULTI_RVO_SIMULATOR_H_

#include "RVOSimulator.h"

namespace RVO {

class RVOMultiSimulator {
 public:
  RVOMultiSimulator(size_t batch);
  RVOMultiSimulator(size_t batch,double timeStep, double neighborDist, size_t maxNeighbors,
                    double timeHorizon, double timeHorizonObst, double radius,
                    double maxSpeed, const Vector2& velocity = Vector2());
  ~RVOMultiSimulator();
  size_t addAgent(const std::vector<Vector2>& position);
  size_t addAgentCar(const std::vector<Vector2>& position);
  void setCarLookupTable();
  size_t addAgent(const std::vector<Vector2>& position, double neighborDist,
                  size_t maxNeighbors, double timeHorizon,
                  double timeHorizonObst, double radius, double maxSpeed,
                  const Vector2& velocity = Vector2());
  size_t addAgentCar(const std::vector<Vector2>& position, double neighborDist,
                     size_t maxNeighbors, double timeHorizon,
                     double timeHorizonObst, double radius, double maxSpeed,
                     const Vector2& velocity = Vector2(), double lambda = 0.5f);
  size_t addObstacle(const std::vector<Vector2>& vertices);
  void clearObstacle();
  void doStep();
  void computeAgents();
  void updateAgents();
  void doStepCar();
  void setNewtonParameters(size_t maxIter, double tol, double d0, double coef, double alphaMin);
  void doNewtonStep(bool require_grad);
  const std::vector<Eigen::MatrixXd>& getGradV();
  const std::vector<Eigen::MatrixXd>& getGradX();
  size_t getAgentMaxNeighbors(size_t agentNo) const;
  double getAgentCarTheta(size_t agentNo) const;
  double getAgentCarPhi(size_t agentNo) const;
  double getAgentMaxSpeed(size_t agentNo) const;
  const std::vector<Vector2>& getAgentPosition(size_t agentNo);
  const std::vector<Vector2>& getAgentPrefVelocity(size_t agentNo);
  double getAgentRadius(size_t agentNo) const;
  const std::vector<Vector2>& getAgentVelocity(size_t agentNo);
  const std::vector<Vector2>& getAgentNewVelocity(size_t agentNo);
  double getGlobalTime() const;
  size_t getNumAgents() const;
  size_t getNumAgentCars() const;
  size_t getNumObstacleVertices() const;
  double getTimeStep() const;
  void processObstacles();
  bool queryVisibility(const Vector2& point1, const Vector2& point2,
                       double radius = 0.0) const;
  void setAgentDefaults(double neighborDist, size_t maxNeighbors,
                        double timeHorizon, double timeHorizonObst,
                        double radius, double maxSpeed,
                        const Vector2& velocity = Vector2());
  void setAgentLambda(size_t agentNo, double lambda);
  void setAgentMaxNeighbors(size_t agentNo, size_t maxNeighbors);
  void setAgentMaxSpeed(size_t agentNo, double maxSpeed);
  void setAgentPosition(size_t agentNo, const std::vector<Vector2>& position);
  void setAgentPrefVelocity(size_t agentNo, const std::vector<Vector2>& prefVelocity);
  void setAgentRadius(size_t agentNo, double radius);
  void setAgentVelocity(size_t agentNo, const std::vector<Vector2>& velocity);
  void setCarProperties(double length, double radius, double vDrivingMax, double vSteeringMax, double aDrivingMax, double phiMax, double dtc, double errorPreferred, double ka, double kv, double kp, double deltaV, double deltaPhi);
  void setTimeStep(double timeStep);
  void setAgentCar(size_t agentCarNo, const std::vector<Vector2>& position);
 private:
  RVOMultiSimulator operator=(const RVOMultiSimulator&) const {
    //diable copy constructor
    return RVOMultiSimulator(_sims.size());
  }
  std::vector<RVOSimulator*> _sims;
  std::vector<Eigen::MatrixXd> _gradVs,_gradXs;
  std::vector<Vector2> _positions,_perfVelocities,_agentVelocities,_agentNewVelocities;
};
}

#endif

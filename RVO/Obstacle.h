#ifndef OBSTACLE_H
#define OBSTACLE_H

#include "Agent.h"

namespace RVO {
class Obstacle {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  Obstacle() {}
  Obstacle(const Vec2T& pos):_pos(pos) {}
  std::shared_ptr<Obstacle> _next;
  Vec2T _pos;
};
class AgentObstacleNeighbor {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  bool operator<(const AgentObstacleNeighbor& other) const;
  bool operator==(const AgentObstacleNeighbor& other) const;
  std::shared_ptr<Agent> _v;
  std::shared_ptr<Obstacle> _o;
};
}

#endif

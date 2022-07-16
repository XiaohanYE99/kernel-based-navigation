#ifndef AGENT_H
#define AGENT_H

#include "Pragma.h"
#include <Eigen/Dense>
#include <memory>

namespace RVO {
class Agent {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  Agent(int id);
  Vec2T operator()(VecCM pos) const;
  int _id;
};
class AgentNeighbor {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  bool operator<(const AgentNeighbor& other) const;
  bool operator==(const AgentNeighbor& other) const;
  std::shared_ptr<Agent> _v[2];
};
}

#endif

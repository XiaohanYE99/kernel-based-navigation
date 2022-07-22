#include "Agent.h"

namespace RVO {
Agent::Agent(int id):_id(id) {}
Agent::Vec2T Agent::operator()(VecCM pos) const {
  return pos.template segment<2>(_id*2);
}
bool AgentNeighbor::operator<(const AgentNeighbor& other) const {
  if(_v[0]<other._v[0])
    return true;
  if(_v[0]>other._v[0])
    return false;
  if(_v[1]<other._v[1])
    return true;
  if(_v[1]>other._v[1])
    return false;
  return false;
}
bool AgentNeighbor::operator==(const AgentNeighbor& other) const {
  return _v[0]==other._v[0] && _v[1]==other._v[1];
}
}

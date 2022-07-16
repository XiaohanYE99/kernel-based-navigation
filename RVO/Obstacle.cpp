#include "Obstacle.h"

namespace RVO {
bool AgentObstacleNeighbor::operator<(const AgentObstacleNeighbor& other) const {
  if(_v<other._v)
    return true;
  if(_v>other._v)
    return false;
  if(_o<other._o)
    return true;
  if(_o>other._o)
    return false;
  return false;
}
bool AgentObstacleNeighbor::operator==(const AgentObstacleNeighbor& other) const {
  return _v==other._v && _o==other._o;
}
}

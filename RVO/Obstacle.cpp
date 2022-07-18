#include "Obstacle.h"

namespace RVO {
Obstacle::Obstacle():_id(-1) {}
Obstacle::Obstacle(const Vec2T& pos,int id):_pos(pos),_id(id) {}
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

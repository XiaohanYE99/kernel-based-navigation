#include "SpatialHash.h"

namespace RVO {
const std::vector<std::shared_ptr<Agent>>& SpatialHash::vss() const {
  return _vss;
}
void SpatialHash::addVertex(std::shared_ptr<Agent> v) {
  _vss.push_back(v);
}
}

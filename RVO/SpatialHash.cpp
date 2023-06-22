#include "SpatialHash.h"

namespace RVO {
SpatialHash::SpatialHash() {}
const std::vector<std::shared_ptr<Agent>>& SpatialHash::vss() const {
  return _vss;
}
void SpatialHash::addVertex(std::shared_ptr<Agent> v) {
  _vss.push_back(v);
}
void SpatialHash::addVertex() {
  if(!_backup.empty()) {
    _backup.back()->_id=(int)_vss.size();
    _vss.push_back(_backup.back());
    _backup.pop_back();
  } else {
    _vss.push_back(std::shared_ptr<Agent>(new Agent((int)_vss.size())));
  }
}
void SpatialHash::clearVertex() {
  _backup.insert(_backup.end(),_vss.begin(),_vss.end());
  _vss.clear();
}
void SpatialHash::removeVertex() {
  _backup.push_back(_vss.back());
  _vss.pop_back();
}
//helper
SpatialHash::SpatialHash(const SpatialHash&) {}
SpatialHash& SpatialHash::operator=(const SpatialHash&) {
  return *this;
}
}

#include "MultiCoverageEnergy.h"

namespace RVO {
MultiCoverageEnergy::MultiCoverageEnergy(const MultiRVOSimulator& sim,T range,bool visibleOnly) {
  _coverage.resize(sim.getBatchSize());
  for(int id=0; id<(int)_coverage.size(); id++)
    _coverage[id].reset(new CoverageEnergy(sim.getSubSimulator(id),range,visibleOnly));
}
std::vector<MultiCoverageEnergy::T> MultiCoverageEnergy::loss(std::vector<Vec> pos) {
  std::vector<T> ret((int)_coverage.size());
  OMP_PARALLEL_FOR_
  for(int id=0; id<(int)_coverage.size(); id++)
    ret[id]=_coverage[id]->loss(pos[id]);
  return ret;
}
std::vector<MultiCoverageEnergy::Vec> MultiCoverageEnergy::grad() const {
  std::vector<Vec> ret((int)_coverage.size());
  OMP_PARALLEL_FOR_
  for(int id=0; id<(int)_coverage.size(); id++)
    ret[id]=_coverage[id]->grad();
  return ret;
}
}

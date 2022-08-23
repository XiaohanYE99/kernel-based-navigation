#ifndef MULTI_COVERAGE_ENERGY_H
#define MULTI_COVERAGE_ENERGY_H

#include "CoverageEnergy.h"
#include "MultiRVO.h"

namespace RVO {
class MultiCoverageEnergy {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  MultiCoverageEnergy(const MultiRVOSimulator& sim,T range,bool visibleOnly=true);
#ifndef SWIG
  std::vector<T> loss(std::vector<Vec> pos);
  std::vector<Vec> grad() const;
#else
  std::vector<double> loss(std::vector<Eigen::Matrix<double,-1,1>> pos);
  std::vector<Eigen::Matrix<double,-1,1>> grad() const;
#endif
 private:
  std::vector<std::shared_ptr<CoverageEnergy>> _coverage;
};
}

#endif

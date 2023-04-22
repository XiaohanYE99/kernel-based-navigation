#ifndef MULTI_VISIBILITY_H
#define MULTI_VISIBILITY_H

#include "Visibility.h"
#include "MultiRVO.h"

namespace RVO {
class MultiVisibilityGraph : protected VisibilityGraph {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  MultiVisibilityGraph(RVOSimulator& rvo);
  MultiVisibilityGraph(MultiRVOSimulator& rvo);
#ifdef SWIG
  void setAgentTargets(const std::vector<Eigen::Matrix<double,2,1>> target,T maxVelocity);
  std::vector<Eigen::Matrix<double,2,1>> setAgentPositions(const std::vector<Eigen::Matrix<double,2,1>> positions);
  std::vector<Eigen::Matrix<double,2,2>> getAgentDVDPs() const;
#else
  void setAgentTargets(const std::vector<Vec2T>& target,T maxVelocity);
  std::vector<Vec2T> setAgentPositions(const std::vector<Vec2T>& positions);
  std::vector<Mat2T> getAgentDVDPs() const;
#endif
 protected:
  std::vector<ShortestPath> _pathVec;
};
}

#endif

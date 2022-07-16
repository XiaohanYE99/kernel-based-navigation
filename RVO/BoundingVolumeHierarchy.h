#ifndef BOUNDING_VOLUME_HIERARCHY_H
#define BOUNDING_VOLUME_HIERARCHY_H

#include "Obstacle.h"
#include "BVHNode.h"

namespace RVO {
class BoundingVolumeHierarchy {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  void clearObstacle();
  void addObstacle(const std::vector<Vec2T>& vss);
  const std::vector<std::shared_ptr<Obstacle>>& getObstacles() const;
  const std::vector<Node<int,BBox>>& getNodes() const;
 private:
  void assemble();
  std::vector<std::shared_ptr<Obstacle>> _obs;
  std::vector<Node<int,BBox>> _bvh;
};
}

#endif

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
  int getNrObstacle() const;
  std::vector<Vec2T> getObstacle(int i) const;
  int getNrVertex() const;
  std::shared_ptr<Obstacle> getVertex(int i) const;
  void addObstacle(const std::vector<Vec2T>& vss);
  const std::vector<std::shared_ptr<Obstacle>>& getObstacles() const;
  const std::vector<Node<int,BBox>>& getNodes() const;
  bool visible(const Vec2T& a,const Vec2T& b,std::shared_ptr<Obstacle> obs=NULL) const;
  bool visible(const Vec2T& a,std::shared_ptr<Obstacle> o,Vec2T* b=NULL) const;
  static bool intersect(const Vec2T edgeA[2],const Vec2T edgeB[2]);
 private:
  void assemble();
  std::vector<std::shared_ptr<Obstacle>> _obs;
  std::vector<Node<int,BBox>> _bvh;
};
}

#endif

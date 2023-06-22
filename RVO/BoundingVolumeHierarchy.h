#ifndef BOUNDING_VOLUME_HIERARCHY_H
#define BOUNDING_VOLUME_HIERARCHY_H

#include "Obstacle.h"
#include "BVHNode.h"

namespace RVO {
class BoundingVolumeHierarchy {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_I
  DECL_MAT_VEC_MAP_TYPES_T
  BoundingVolumeHierarchy();
  BoundingVolumeHierarchy(const BoundingVolumeHierarchy& other,bool simplify=false);
  virtual ~BoundingVolumeHierarchy();
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
  static T distanceAgentObstacle(const Vec2T edgeA[2],const Vec2T edgeB[2]);
  static T distanceAgentAgent(const Vec2T edgeA[2],const Vec2T edgeB[2]);
  static T distance(const Vec2T& pt,const Vec2T edgeB[2]);
  static T closestT(const Vec2T& pt,const Vec2T edgeB[2]);
 private:
  void assembleFull();
  void assembleSimplified();
  template <typename T2>
  void addObstacleInternal(const std::vector<Eigen::Matrix<T2,2,1>>& vss);
  template <typename T2>
  static void removeDuplicateVertices(std::vector<Eigen::Matrix<T2,2,1>>& vss);
  std::vector<std::shared_ptr<Obstacle>> _obs,_backup;
  std::vector<Node<int,BBox>> _bvh;
};
}

#endif

#ifndef BBOX_H
#define BBOX_H

#include "Pragma.h"
#include <Eigen/Dense>

namespace RVO {
struct BBox {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  BBox();
  BBox(const Vec2T& p);
  BBox(const Vec2T& minC,const Vec2T& maxC);
  BBox(const BBox& other);
  virtual ~BBox();
  template <typename T2>
  BBox& operator=(const BBox& other) {
    copy(other);
    return *this;
  }
  static BBox createMM(const Vec2T& minC,const Vec2T& maxC);
  static BBox createME(const Vec2T& minC,const Vec2T& extent);
  static BBox createCE(const Vec2T& center,const Vec2T& extent);
  BBox getIntersect(const BBox& other) const;
  BBox getUnion(const BBox& other) const;
  BBox getUnion(const Vec2T& point) const;
  BBox getUnion(const Vec2T& ctr,const T& rad) const;
  void setIntersect(const BBox& other);
  void setUnion(const BBox& other);
  void setUnion(const Vec2T& point);
  void setUnion(const Vec2T& ctr,const T& rad);
  void setPoints(const Vec2T& a,const Vec2T& b,const Vec2T& c);
  void setPoints(const Vec2T& a,const Vec2T& b,const Vec2T& c,const Vec2T& d);
  const Vec2T& minCorner() const;
  const Vec2T& maxCorner() const;
  void enlargedEps(T eps);
  BBox enlargeEps(T eps) const;
  void enlarged(T len);
  BBox enlarge(T len) const;
  Vec2T lerp(const Vec2T& frac) const;
  bool empty() const;
  template <int DIM2>
  bool containDim(const Vec2T& point) const {
    for(int i=0; i<DIM2; i++)
      if(_minC[i] > point[i] || _maxC[i] < point[i])
        return false;
    return true;
  }
  bool contain(const BBox& other) const;
  bool contain(const Vec2T& point) const;
  bool contain(const Vec2T& point,const T& rad) const;
  void reset();
  Vec2T getExtent() const;
  T distTo(const BBox& other) const;
  T distTo(const Vec2T& pt) const;
  T distToSqr(const Vec2T& pt) const;
  Vec2T closestTo(const Vec2T& pt) const;
  bool intersect(const Vec2T& p,const Vec2T& q) const;
  bool intersect(const Vec2T& p,const Vec2T& q,T& s,T& t) const;
  bool intersect(const BBox& other) const;
  Vec2T project(const Vec2T& a) const;
  BBox& copy(const BBox& other);
  T perimeter() const;
  Vec2T _minC,_maxC;
};
}

#endif

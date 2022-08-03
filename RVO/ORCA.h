#ifndef ORCA_H
#define ORCA_H

#include "RVO.h"

namespace RVO {
class ORCASimulator : public RVOSimulator {
 public:
  struct VelocityObstacle {
    Eigen::Map<Vec2T> pos() {
      return Eigen::Map<Vec2T>(_vnab.data());
    }
    Eigen::Map<const Vec2T> pos() const {
      return Eigen::Map<const Vec2T>(_vnab.data());
    }
    Eigen::Map<Vec2T> nor() {
      return Eigen::Map<Vec2T>(_vnab.data()+2);
    }
    Eigen::Map<const Vec2T> nor() const {
      return Eigen::Map<const Vec2T>(_vnab.data()+2);
    }
    Vec2T proj(const Vec2T& v) const {
      return v-(v-pos()).dot(nor())*nor();
    }
    bool outside(const Vec2T& v) const {
      return (v-pos()).dot(nor())>0;
    }
    Vec4T _vnab;    //_vab is a point on the boundary, _nab is the outward-pointing normal
    Mat4T _DvnDpva;
    Mat2T _DvnDpvb;
    int _aid,_bid;
    int _case;
  };
  ORCASimulator(const ORCASimulator& other);
  ORCASimulator& operator=(const ORCASimulator& other);
  ORCASimulator(T rad,T d0=1,T gTol=1e-4,T coef=1,T timestep=1,int maxIter=1000,bool radixSort=false,bool useHash=true);
  virtual bool optimize(bool requireGrad,bool output) override;
  void debugVO(int aid,int bid,int testCase,T eps=1e-4f);
  void debugVO(int aid,int testCase,bool mustInside,T eps=1e-4f);
 protected:
  VelocityObstacle computeVelocityObstacle(int aid,int bid,bool requireGrad) const;
  VelocityObstacle computeVelocityObstacle(int aid,Vec2T o[2],bool requireGrad) const;
  void computeVelocityObstacle(VelocityObstacle& ret,const Vec2T& o,const Vec2T& dL,const Vec2T& dR,bool requireGrad) const;
  void computeVelocityObstacle(VelocityObstacle& ret,const Vec2T o[2],const Vec2T& dL,const Vec2T& dR,bool requireGrad) const;
  void updateLP(const VelocityObstacle& VO) const {}
  static T cross2D(const Vec2T& a,const Vec2T& b);
  static Mat2T rot2D(T dist,T cord);
};
}

#endif

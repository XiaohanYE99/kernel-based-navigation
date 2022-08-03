#ifndef ORCA_H
#define ORCA_H

#include "RVO.h"
#include <unsupported/Eigen/AutoDiff>

namespace RVO {
class ORCASimulator : public RVOSimulator {
 public:
  typedef Eigen::Matrix<T,8,1> Derivative;
  typedef Eigen::AutoDiffScalar<Derivative> AD;
  typedef Eigen::Matrix<AD,2,2> Mat2TAD;
  typedef Eigen::Matrix<AD,2,1> Vec2TAD;
  struct VelocityObstacle {
    Vec2T proj(const Vec2T& v) const {
      return v-(v-pos()).dot(nor())*nor();
    }
    bool outside(const Vec2T& v) const {
      return (v-pos()).dot(nor())>0;
    }
    Vec2T pos() const {
      return Vec2T(_pos[0].value(),_pos[1].value());
    }
    Vec2T nor() const {
      return Vec2T(_nor[0].value(),_nor[1].value());
    }
    Mat2T DposDpa() const {
      Mat2T ret;
      ASSERT(_aid>=0)
      ret.row(0)=_pos[0].derivatives().template segment<2>(0);
      ret.row(1)=_pos[1].derivatives().template segment<2>(0);
      return ret;
    }
    Mat2T DposDpb() const {
      Mat2T ret;
      ASSERT(_bid>=0)
      ret.row(0)=_pos[0].derivatives().template segment<2>(2);
      ret.row(1)=_pos[1].derivatives().template segment<2>(2);
      return ret;
    }
    Mat2T DposDva() const {
      Mat2T ret;
      ASSERT(_aid>=0)
      ret.row(0)=_pos[0].derivatives().template segment<2>(4);
      ret.row(1)=_pos[1].derivatives().template segment<2>(4);
      return ret;
    }
    Mat2T DposDvb() const {
      Mat2T ret;
      ASSERT(_bid>=0)
      ret.row(0)=_pos[0].derivatives().template segment<2>(6);
      ret.row(1)=_pos[1].derivatives().template segment<2>(6);
      return ret;
    }
    Mat2T DnorDpa() const {
      Mat2T ret;
      ASSERT(_aid>=0)
      ret.row(0)=_nor[0].derivatives().template segment<2>(0);
      ret.row(1)=_nor[1].derivatives().template segment<2>(0);
      return ret;
    }
    Mat2T DnorDpb() const {
      Mat2T ret;
      ASSERT(_bid>=0)
      ret.row(0)=_nor[0].derivatives().template segment<2>(2);
      ret.row(1)=_nor[1].derivatives().template segment<2>(2);
      return ret;
    }
    Mat2T DnorDva() const {
      Mat2T ret;
      ASSERT(_aid>=0)
      ret.row(0)=_nor[0].derivatives().template segment<2>(4);
      ret.row(1)=_nor[1].derivatives().template segment<2>(4);
      return ret;
    }
    Mat2T DnorDvb() const {
      Mat2T ret;
      ASSERT(_bid>=0)
      ret.row(0)=_nor[0].derivatives().template segment<2>(6);
      ret.row(1)=_nor[1].derivatives().template segment<2>(6);
      return ret;
    }
    Vec2TAD _pos,_nor;    //_vab is a point on the boundary, _nab is the outward-pointing normal
    int _aid,_bid;
    int _case;
  };
  ORCASimulator(const ORCASimulator& other);
  ORCASimulator& operator=(const ORCASimulator& other);
  ORCASimulator(T rad,T d0=1,T gTol=1e-4,T coef=1,T timestep=1,int maxIter=1000,bool radixSort=false,bool useHash=true);
  virtual bool optimize(bool requireGrad,bool output) override;
  void debugVO(int aid,int bid,int testCase,T eps=1e-4f);
  void debugVO(int aid,int testCase,bool mustInside,T eps=1e-4f);
  void debugDerivatives(const VelocityObstacle& VO);
  void debugDerivatives(const VelocityObstacle& VO,const Vec2T o[2]);
 protected:
  VelocityObstacle computeVelocityObstacle(int aid,int bid) const;
  VelocityObstacle computeVelocityObstacle(int aid,const Vec2T o[2]) const;
  void computeVelocityObstacle(VelocityObstacle& ret,const Vec2TAD& o,const Vec2TAD& a,const Vec2TAD& dL,const Vec2TAD& dR) const;
  void computeVelocityObstacle(VelocityObstacle& ret,const Vec2TAD o[2],const Vec2TAD a[2],const Vec2TAD& dL,const Vec2TAD& dR) const;
  void updateLP(const VelocityObstacle& VO) const;
  static T cross2D(const Vec2TAD& a,const Vec2TAD& b);
  static T cross2D(const Vec2T& a,const Vec2T& b);
  static Mat2TAD rot2D(AD dist,AD cord);
};
}

#endif

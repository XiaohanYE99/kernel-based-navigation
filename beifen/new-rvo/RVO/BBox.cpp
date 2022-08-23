#include "BBox.h"
#include "Epsilon.h"

namespace RVO {
//BBox
BBox::BBox() {
  reset();
}
BBox::BBox(const Vec2T& p):_minC(p),_maxC(p) {}
BBox::BBox(const Vec2T& minC,const Vec2T& maxC):_minC(minC),_maxC(maxC) {}
BBox::BBox(const BBox& other) {
  copy(other);
}
BBox::~BBox() {}
BBox BBox::createMM(const Vec2T& minC,const Vec2T& maxC) {
  return BBox(minC,maxC);
}
BBox BBox::createME(const Vec2T& minC,const Vec2T& extent) {
  return BBox(minC,minC+extent);
}
BBox BBox::createCE(const Vec2T& center,const Vec2T& extent) {
  return BBox(center-extent,center+extent);
}
BBox BBox::getIntersect(const BBox& other) const {
  return createMM(_minC.cwiseMax(other._minC),_maxC.cwiseMin(other._maxC));
}
BBox BBox::getUnion(const BBox& other) const {
  return createMM(_minC.cwiseMin(other._minC),_maxC.cwiseMax(other._maxC));
}
BBox BBox::getUnion(const Vec2T& point) const {
  return createMM(_minC.cwiseMin(point),_maxC.cwiseMax(point));
}
BBox BBox::getUnion(const Vec2T& ctr,const T& rad) const {
  return createMM(_minC.cwiseMin(ctr-Vec2T::Constant(rad)),_maxC.cwiseMax(ctr+Vec2T::Constant(rad)));
}
void BBox::setIntersect(const BBox& other) {
  _minC=_minC.cwiseMax(other._minC);
  _maxC=_maxC.cwiseMin(other._maxC);
}
void BBox::setUnion(const BBox& other) {
  _minC=_minC.cwiseMin(other._minC);
  _maxC=_maxC.cwiseMax(other._maxC);
}
void BBox::setUnion(const Vec2T& point) {
  _minC=_minC.cwiseMin(point);
  _maxC=_maxC.cwiseMax(point);
}
void BBox::setUnion(const Vec2T& ctr,const T& rad) {
  _minC=_minC.cwiseMin(ctr-Vec2T::Constant(rad));
  _maxC=_maxC.cwiseMax(ctr+Vec2T::Constant(rad));
}
void BBox::setPoints(const Vec2T& a,const Vec2T& b,const Vec2T& c) {
  _minC=a.cwiseMin(b).cwiseMin(c);
  _maxC=a.cwiseMax(b).cwiseMax(c);
}
void BBox::setPoints(const Vec2T& a,const Vec2T& b,const Vec2T& c,const Vec2T& d) {
  _minC=a.cwiseMin(b).cwiseMin(c).cwiseMin(d);
  _maxC=a.cwiseMax(b).cwiseMax(c).cwiseMax(d);
}
const BBox::Vec2T& BBox::minCorner() const {
  return _minC;
}
const BBox::Vec2T& BBox::maxCorner() const {
  return _maxC;
}
void BBox::enlargedEps(T eps) {
  Vec2T d=(_maxC-_minC)*T(eps*0.5f);
  _minC-=d;
  _maxC+=d;
}
BBox BBox::enlargeEps(T eps) const {
  Vec2T d=(_maxC-_minC)*T(eps*0.5f);
  return createMM(_minC-d,_maxC+d);
}
void BBox::enlarged(T len) {
  for(int i=0; i<2; i++) {
    _minC[i]-=len;
    _maxC[i]+=len;
  }
}
BBox BBox::enlarge(T len) const {
  BBox ret=createMM(_minC,_maxC);
  ret.enlarged(len);
  return ret;
}
typename BBox::Vec2T BBox::lerp(const Vec2T& frac) const {
  return (_maxC.array()*frac.array()-_minC.array()*(frac.array()-T(1))).matrix();
}
bool BBox::empty() const {
  return (_minC.array()>=_maxC.array()).any();
}
bool BBox::contain(const BBox& other) const {
  for(int i=0; i<2; i++)
    if(_minC[i] > other._minC[i] || _maxC[i] < other._maxC[i])
      return false;
  return true;
}
bool BBox::contain(const Vec2T& point) const {
  for(int i=0; i<2; i++)
    if(_minC[i] > point[i] || _maxC[i] < point[i])
      return false;
  return true;
}
bool BBox::contain(const Vec2T& point,const T& rad) const {
  for(int i=0; i<2; i++)
    if(_minC[i]+rad > point[i] || _maxC[i]-rad < point[i])
      return false;
  return true;
}
void BBox::reset() {
  _minC=Vec2T::Constant( std::numeric_limits<double>::max());
  _maxC=Vec2T::Constant(-std::numeric_limits<double>::max());
}
BBox::Vec2T BBox::getExtent() const {
  return _maxC-_minC;
}
BBox::T BBox::distTo(const BBox& other) const {
  Vec2T dist=Vec2T::Zero();
  for(int i=0; i<2; i++) {
    if (other._maxC[i] < _minC[i])
      dist[i] = other._maxC[i] - _minC[i];
    else if (other._minC[i] > _maxC[i])
      dist[i] = other._minC[i] - _maxC[i];
  }
  return sqrt(dist.squaredNorm());
}
BBox::T BBox::distTo(const Vec2T& p) const {
  return sqrt(distToSqr(p));
}
BBox::T BBox::distToSqr(const Vec2T& p) const {
  Vec2T dist=Vec2T::Zero();
  for(int i=0; i<2; i++) {
    if (p[i] < _minC[i])
      dist[i] = p[i] - _minC[i];
    else if (p[i] > _maxC[i])
      dist[i] = p[i] - _maxC[i];
  }
  return dist.squaredNorm();
}
BBox::Vec2T BBox::closestTo(const Vec2T& p) const {
  Vec2T dist(p);
  for(int i=0; i<2; i++) {
    if (p[i] < _minC[i])
      dist[i] = _minC[i];
    else if (p[i] > _maxC[i])
      dist[i] = _maxC[i];
  }
  return dist;
}
bool BBox::intersect(const Vec2T& p,const Vec2T& q) const {
  T s=0, t=1;
  return intersect(p,q,s,t);
}
bool BBox::intersect(const Vec2T& p,const Vec2T& q,T& s,T& t) const {
  const T lo=1-Epsilon<T>::defaultEps();
  const T hi=1+Epsilon<T>::defaultEps();

  s=0;
  t=1;
  for(int i=0; i<2; ++i) {
    T D=q[i]-p[i];
    if(p[i]<q[i]) {
      T s0=lo*(_minC[i]-p[i])/D, t0=hi*(_maxC[i]-p[i])/D;
      if(s0>s) s=s0;
      if(t0<t) t=t0;
    } else if(p[i]>q[i]) {
      T s0=lo*(_maxC[i]-p[i])/D, t0=hi*(_minC[i]-p[i])/D;
      if(s0>s) s=s0;
      if(t0<t) t=t0;
    } else {
      if(p[i]<_minC[i] || p[i]>_maxC[i])
        return false;
    }

    if(s>t)
      return false;
  }
  return true;
}
bool BBox::intersect(const BBox& other) const {
  for(int i=0; i<2; i++)
    if(_maxC[i] < other._minC[i] || other._maxC[i] < _minC[i])
      return false;
  return true;
  //return compLE(_minC,other._maxC) && compLE(other._minC,_maxC);
}
BBox::Vec2T BBox::project(const Vec2T& a) const {
  Vec2T ctr=(_minC+_maxC)*0.5f;
  T ctrD=a.dot(ctr);
  T delta=0.0f;
  ctr=_maxC-ctr;
  for(int i=0; i<2; i++)
    delta+=fabs(ctr[i]*a[i]);
  return Vec2T(ctrD-delta,ctrD+delta);
}
BBox& BBox::copy(const BBox& other) {
  for(int i=0; i<2; i++) {
    _minC[i]=other._minC[i];
    _maxC[i]=other._maxC[i];
  }
  return *this;
}
BBox::T BBox::perimeter() const {
  Vec2T ext=getExtent();
  return (ext[0]+ext[1])*2.0f;
}
}

#ifndef ZERO_H
#define ZERO_H

#include <Eigen/Dense>

namespace RVO {
template <typename T2>
struct Zero {
  static T2 value() {
    return (T2)0.0f;
  }
  static T2 value(const T2&) {
    return (T2)0.0f;
  }
};
template <typename T2,int r,int c,int o,int mr,int mc>
struct Zero<Eigen::Matrix<T2,r,c,o,mr,mc> > {
  typedef Eigen::Matrix<T2,r,c,o,mr,mc> ValueType;
  static ValueType value() {
    ValueType ret;
    ret.setZero();
    return ret;
  }
  static ValueType value(const ValueType& ref) {
    ValueType ret;
    ret.resize(ref.rows(),ref.cols());
    ret.setZero();
    return ret;
  }
};
}

#endif

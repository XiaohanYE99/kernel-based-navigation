#ifndef ZERO_H
#define ZERO_H

#include <Eigen/Dense>

namespace RVO {
//ReduceAdd
template <typename T>
struct ReduceAdd {
  static T init(const T&) {
    return (T)0.0f;
  }
  static T reduce(const T& a,const T& b) {
    return a+b;
  }
};
template <typename T,int r,int c,int o,int mr,int mc>
struct ReduceAdd<Eigen::Matrix<T,r,c,o,mr,mc> > {
  typedef Eigen::Matrix<T,r,c,o,mr,mc> ValueType;
  static ValueType init(const ValueType& ref) {
    return ValueType::Zero(ref.rows(),ref.cols());
  }
  static ValueType reduce(const ValueType& a,const ValueType& b) {
    return a+b;
  }
};
//ReduceMin/Max
template <typename T>
struct ReduceMin {
  static T init(const T&) {
    return (T)std::numeric_limits<double>::max();
  }
  static T reduce(const T& a,const T& b) {
    return std::min(a,b);
  }
};
template <typename T>
struct ReduceMax {
  static T init(const T&) {
    return (T)-std::numeric_limits<double>::max();
  }
  static T reduce(const T& a,const T& b) {
    return std::max(a,b);
  }
};
template <typename T,int r,int c,int o,int mr,int mc>
struct ReduceMin<Eigen::Matrix<T,r,c,o,mr,mc> > {
  typedef Eigen::Matrix<T,r,c,o,mr,mc> ValueType;
  static ValueType init(const ValueType& ref) {
    return ValueType::Constant(ref.rows(),ref.cols(),std::numeric_limits<double>::max());
  }
  static ValueType reduce(const ValueType& a,const ValueType& b) {
    return a.cwiseMin(b);
  }
};
template <typename T,int r,int c,int o,int mr,int mc>
struct ReduceMax<Eigen::Matrix<T,r,c,o,mr,mc> > {
  typedef Eigen::Matrix<T,r,c,o,mr,mc> ValueType;
  static ValueType init(const ValueType& ref) {
    return ValueType::Constant(ref.rows(),ref.cols(),-std::numeric_limits<double>::max());
  }
  static ValueType reduce(const ValueType& a,const ValueType& b) {
    return a.cwiseMax(b);
  }
};
//ReduceMin/MaxLabel
template <typename T,int r,int c,int o,int mr,int mc>
struct ReduceMin<Eigen::Matrix<std::pair<T,int>,r,c,o,mr,mc>> {
  typedef Eigen::Matrix<std::pair<T,int>,r,c,o,mr,mc> ValueType;
  static ValueType init(const ValueType& ref) {
    return ValueType::Constant(ref.rows(),ref.cols(),std::make_pair(std::numeric_limits<double>::max(),-1));
  }
  static ValueType reduce(const ValueType& a,const ValueType& b) {
    return a.binaryExpr(b,[&](auto va,auto vb) {
      return va.first<vb.first?va:vb;
    });
  }
};
template <typename T,int r,int c,int o,int mr,int mc>
struct ReduceMax<Eigen::Matrix<std::pair<T,int>,r,c,o,mr,mc>> {
  typedef Eigen::Matrix<std::pair<T,int>,r,c,o,mr,mc> ValueType;
  static ValueType init(const ValueType& ref) {
    return ValueType::Constant(ref.rows(),ref.cols(),std::make_pair(-std::numeric_limits<double>::max(),-1));
  }
  static ValueType reduce(const ValueType& a,const ValueType& b) {
    return a.binaryExpr(b,[&](auto va,auto vb) {
      return va.first>vb.first?va:vb;
    });
  }
};
}

#endif

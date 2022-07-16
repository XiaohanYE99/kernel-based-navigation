#ifndef EPSILON_H
#define EPSILON_H

#include <float.h>
#ifdef WITH_QUADMATH
#include <boost/multiprecision/float128.hpp>
#endif
#ifdef WITH_MPFR
#include <boost/multiprecision/mpfr.hpp>
#endif

namespace RVO {
#ifdef WITH_QUADMATH
typedef boost::multiprecision::float128 float128;
#endif
#ifdef WITH_MPFR
typedef boost::multiprecision::mpfr_float mpfr_float;
#endif
template <typename T>
struct Epsilon {
  static T defaultEps() {
    return _defaultEps;
  }
  static T rotationEps() {
    return _rotationEps;
  }
  static T finiteDifferenceEps() {
    return _finiteDifferenceEps;
  }
  static void setDefaultEps(T defaultEps) {
    _defaultEps=defaultEps;
  }
  static void setRotationEps(T rotationEps) {
    _rotationEps=rotationEps;
  }
  static void setFiniteDifferenceEps(T finiteDifferenceEps) {
    _finiteDifferenceEps=finiteDifferenceEps;
  }
 private:
  static T _defaultEps;
  static T _rotationEps;
  static T _finiteDifferenceEps;
};
#ifdef WITH_MPFR
template <>
struct Epsilon<mpfr_float> {
  static mpfr_float defaultEps();
  static mpfr_float rotationEps();
  static mpfr_float finiteDifferenceEps();
  static void setDefaultEps(mpfr_float defaultEps);
  static void setRotationEps(mpfr_float rotationEps);
  static void setFiniteDifferenceEps(mpfr_float finiteDifferenceEps);
 private:
  static void initialize();
  static mpfr_float _defaultEps;
  static mpfr_float _rotationEps;
  static mpfr_float _finiteDifferenceEps;
  static bool _initialized;
};
#endif
}

#endif

#include "Epsilon.h"

namespace RVO {
//float
template <> float Epsilon<float>::_defaultEps=1e-5f;
template <> float Epsilon<float>::_rotationEps=1e-5f;
template <> float Epsilon<float>::_finiteDifferenceEps=1e-5f;
//double
template <> double Epsilon<double>::_defaultEps=1e-9;
template <> double Epsilon<double>::_rotationEps=1e-9;
template <> double Epsilon<double>::_finiteDifferenceEps=1e-9;
//float128
#ifdef WITH_QUADMATH
template <> float128 Epsilon<float128>::_defaultEps=1e-15;
template <> float128 Epsilon<float128>::_rotationEps=1e-15;
template <> float128 Epsilon<float128>::_finiteDifferenceEps=1e-15;
#endif
//mpfr_float
#ifdef WITH_MPFR
mpfr_float Epsilon<mpfr_float>::defaultEps() {
  if(!_initialized)
    initialize();
  return _defaultEps;
}
mpfr_float Epsilon<mpfr_float>::rotationEps() {
  if(!_initialized)
    initialize();
  return _rotationEps;
}
mpfr_float Epsilon<mpfr_float>::finiteDifferenceEps() {
  if(!_initialized)
    initialize();
  return _finiteDifferenceEps;
}
void Epsilon<mpfr_float>::setDefaultEps(mpfr_float defaultEps) {
  _defaultEps=defaultEps;
}
void Epsilon<mpfr_float>::setRotationEps(mpfr_float rotationEps) {
  _rotationEps=rotationEps;
}
void Epsilon<mpfr_float>::setFiniteDifferenceEps(mpfr_float finiteDifferenceEps) {
  _finiteDifferenceEps=finiteDifferenceEps;
}
void Epsilon<mpfr_float>::initialize() {
  _defaultEps=pow(std::numeric_limits<mpfr_float>::epsilon(),mpfr_float(.2f));
  _rotationEps=pow(std::numeric_limits<mpfr_float>::epsilon(),mpfr_float(.2f));
  _finiteDifferenceEps=pow(std::numeric_limits<mpfr_float>::epsilon(),mpfr_float(.2f));
  _initialized=true;
}
mpfr_float Epsilon<mpfr_float>::_defaultEps;
mpfr_float Epsilon<mpfr_float>::_rotationEps;
mpfr_float Epsilon<mpfr_float>::_finiteDifferenceEps;
bool Epsilon<mpfr_float>::_initialized=false;
#endif
}

#ifndef PRAGMA_H
#define PRAGMA_H

#include <algorithm>

//openmp macro prepare
#ifdef _MSC_VER
#define STRINGIFY_OMP(X) X
#define PRAGMA __pragma
#else
#define STRINGIFY_OMP(X) #X
#define PRAGMA _Pragma
#endif
#ifndef NO_OPENMP
//openmp convenient functions
#define OMP_PARALLEL_FOR_ PRAGMA(STRINGIFY_OMP(omp parallel for schedule(static)))
#define OMP_PARALLEL_FOR_I(...) PRAGMA(STRINGIFY_OMP(omp parallel for schedule(static) __VA_ARGS__))
#define OMP_PARALLEL_FOR_X(X) PRAGMA(STRINGIFY_OMP(omp parallel for num_threads(X) schedule(static)))
#define OMP_PARALLEL_FOR_XI(X,...) PRAGMA(STRINGIFY_OMP(omp parallel for num_threads(X) schedule(static) __VA_ARGS__))
#define OMP_ADD(...) reduction(+: __VA_ARGS__)
#define OMP_PRI(...) private(__VA_ARGS__)
#define OMP_FPRI(...) firstprivate(__VA_ARGS__)
#define OMP_ATOMIC_ PRAGMA(STRINGIFY_OMP(omp atomic))
#ifdef _MSC_VER
#define OMP_ATOMIC_CAPTURE_ PRAGMA(STRINGIFY_OMP(omp critical))	// VS doesn't support capture, use critical instead
#else
#define OMP_ATOMIC_CAPTURE_ PRAGMA(STRINGIFY_OMP(omp atomic capture))
#endif
#define OMP_CRITICAL_ PRAGMA(STRINGIFY_OMP(omp critical))
#define OMP_FLUSH_(X) PRAGMA(STRINGIFY_OMP(omp flush(X)))
#else
//openmp convenient functions
#define OMP_PARALLEL_FOR_
#define OMP_PARALLEL_FOR_I(...)
#define OMP_PARALLEL_FOR_X(X)
#define OMP_PARALLEL_FOR_XI(X,...)
#define OMP_ADD(...)
#define OMP_PRI(...)
#define OMP_FPRI(...)
#define OMP_ATOMIC_
#define OMP_ATOMIC_CAPTURE_
#define OMP_CRITICAL_
#define OMP_FLUSH_(X)
#endif

//assert
#define ASSERT(var) {do{if(!(var)){exit(EXIT_FAILURE);}}while(0);}
#define ASSERT_MSG(var,msg) {do{if(!(var)){printf(msg);exit(EXIT_FAILURE);}}while(0);}
#define ASSERT_MSGV(var,msg,...) {do{if(!(var)){printf(msg,__VA_ARGS__);exit(EXIT_FAILURE);}}while(0);}
#define FUNCTION_NOT_IMPLEMENTED ASSERT_MSGV(false,"Function \"%s\" not implemented!",__FUNCTION__)

#define DECL_MAT_VEC_MAP_TYPES_I \
typedef Eigen::Matrix<int,-1,1> Veci;\
typedef Eigen::Matrix<int,2,1> Vec2i;\
typedef Eigen::Matrix<int,4,1> Vec4i;\
\
typedef Eigen::Map<Veci> VeciM;\
typedef Eigen::Map<const Veci> VeciCM;

#define DECL_MAT_VEC_MAP_TYPES_T \
typedef Eigen::Matrix<T,-1,1> Vec;\
typedef Eigen::Matrix<T,2,1> Vec2T;\
typedef Eigen::Matrix<T,3,1> Vec3T;\
typedef Eigen::Matrix<T,4,1> Vec4T;\
typedef Eigen::Matrix<T,6,1> Vec6T;\
\
typedef Eigen::Matrix<T,-1,-1> MatT;\
typedef Eigen::Matrix<T,2,-1> Mat2XT;\
typedef Eigen::Matrix<T,2,2> Mat2T;\
\
typedef Eigen::Map<Vec> VecM;\
typedef Eigen::Map<const Vec> VecCM;\
\
typedef Eigen::Map<MatT> MatTM;\
typedef Eigen::Map<const MatT> MatTCM;\
typedef Eigen::Map<Mat2XT> Mat2XTM;\
typedef Eigen::Map<const Mat2XT> Mat2XTCM;

#define DECL_MAP_FUNCS  \
template <typename T2>   \
static inline Eigen::Map<T2> mapM(T2& m) {   \
  return Eigen::Map<T2>(m.data(),m.rows(),m.cols());  \
}   \
template <typename T2>   \
static inline Eigen::Map<T2> mapV(T2& m) {   \
  return Eigen::Map<T2>(m.data(),m.rows());  \
}   \
template <typename T2>   \
static inline Eigen::Map<T2> mapM(T2* m) {   \
  static const int rows=T2::RowsAtCompileTime>0?T2::RowsAtCompileTime:0;  \
  static const int cols=T2::ColsAtCompileTime>0?T2::ColsAtCompileTime:0;  \
  return m?Eigen::Map<T2>(m->data(),m->rows(),m->cols()):Eigen::Map<T2>(NULL,rows,cols);  \
}   \
template <typename T2>   \
static inline Eigen::Map<T2> mapV(T2* m) {   \
  return m?Eigen::Map<T2>(m->data(),m->rows()):Eigen::Map<T2>(NULL,0);  \
}   \
template <typename T2>   \
static inline Eigen::Map<const T2> mapCM(const T2& m) {   \
  return Eigen::Map<const T2>(m.data(),m.rows(),m.cols());  \
}   \
template <typename T2>   \
static inline Eigen::Map<const T2> mapCV(const T2& m) {   \
  return Eigen::Map<const T2>(m.data(),m.rows());  \
}   \
template <typename T2>   \
static inline Eigen::Map<const T2> mapCM(const T2* m) {   \
  static const int rows=T2::RowsAtCompileTime>0?T2::RowsAtCompileTime:0;  \
  static const int cols=T2::ColsAtCompileTime>0?T2::ColsAtCompileTime:0;  \
  return m?Eigen::Map<const T2>(m->data(),m->rows(),m->cols()):Eigen::Map<const T2>(NULL,rows,cols);  \
}   \
template <typename T2>   \
static inline Eigen::Map<const T2> mapCV(const T2* m) {   \
  return m?Eigen::Map<const T2>(m->data(),m->rows()):Eigen::Map<const T2>(NULL,0);  \
}   \
template <typename T2>   \
static inline Eigen::Map<const T2> mapM2CM(Eigen::Map<T2> m) {   \
  return Eigen::Map<const T2>(m.data(),m.rows(),m.cols());  \
}   \
template <typename T2>   \
static inline Eigen::Map<T2> mapCM2M(Eigen::Map<const T2> m) {   \
  return Eigen::Map<T2>(m.data(),m.rows(),m.cols());  \
}   \
template <typename T2>   \
static inline Eigen::Map<const T2> mapV2CV(Eigen::Map<T2> m) {   \
  return Eigen::Map<const T2>(m.data(),m.rows());  \
}   \
template <typename T2>   \
static inline Eigen::Map<T2> mapCV2V(Eigen::Map<const T2> m) {   \
  return Eigen::Map<T2>(m.data(),m.rows());  \
}

#define REUSE_MAP_FUNCS_T(T)  \
using T::mapM;\
using T::mapV;\
using T::mapCM;\
using T::mapCV;\
using T::mapM2CM;\
using T::mapCM2M;\
using T::mapV2CV;\
using T::mapCV2V;

//basic functions
template <typename T>
void sort2(T& a,T& b) {
  if(a>b)
    std::swap(a,b);
}

#endif

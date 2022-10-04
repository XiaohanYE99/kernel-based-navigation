#ifndef LBFGS_UPDATE_H
#define LBFGS_UPDATE_H

#include "Pragma.h"
#include "Epsilon.h"
#include <Eigen/Dense>

namespace RVO {
class LBFGSUpdate {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  DECL_MAP_FUNCS
  LBFGSUpdate();
  //params
  int nCorrect() const;
  void nCorrect(int n);
  void mulHv(VecCM v,VecM res);
  void update(VecCM s,VecCM y);
  void reset(int inputs);
 protected:
  T _theta;
  MatT _y,_s;
  Vec _ys,_alpha;
  int _ptr,_nCorrect,_nCorrectCurr;
};
}

#endif

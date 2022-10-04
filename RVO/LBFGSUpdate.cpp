#include "LBFGSUpdate.h"

namespace RVO {
LBFGSUpdate::LBFGSUpdate():_nCorrect(10) {}
int LBFGSUpdate::nCorrect() const {
  return _nCorrect;
}
void LBFGSUpdate::nCorrect(int n) {
  _nCorrect=n;
}
void LBFGSUpdate::mulHv(VecCM v,VecM res) {
  res=v;
  int j=_ptr%_nCorrect;
  //first loop
  for(int i=0; i<_nCorrectCurr; i++) {
    j=(j+_nCorrect-1)%_nCorrect;
    _alpha[j]=_s.col(j).dot(res)/_ys[j];
    res-=_alpha[j]*_y.col(j);
  }
  //mul H0
  res/=_theta;
  //second loop
  for(int i=0; i<_nCorrectCurr; i++) {
    T beta=_y.col(j).dot(res)/_ys[j];
    res+=(_alpha[j]-beta)*_s.col(j);
    j=(j+1)%_nCorrect;
  }
}
void LBFGSUpdate::update(VecCM s,VecCM y) {
  int loc=_ptr%_nCorrect;
  _s.col(loc)=s;
  _y.col(loc)=y;
  _ys[loc]=y.dot(s);
  _theta=y.squaredNorm()/_ys[loc];
  if(_nCorrectCurr<_nCorrect)
    _nCorrectCurr++;
  _ptr=loc+1;
}
void LBFGSUpdate::reset(int inputs) {
  _y.resize(inputs,_nCorrect);
  _s.resize(inputs,_nCorrect);
  _ys.resize(_nCorrect);
  _alpha.resize(_nCorrect);
  _nCorrectCurr=0;
  _ptr=_nCorrect;
  _theta=1;
}
}

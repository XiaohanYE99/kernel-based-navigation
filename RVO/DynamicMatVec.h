#ifndef MAT_VEC_H
#define MAT_VEC_H

#include "Pragma.h"
#include "Epsilon.h"

namespace RVO {
template <typename T>
struct DynamicMat {
  DECL_MAT_VEC_MAP_TYPES_T
  DECL_MAP_FUNCS
  DynamicMat():_cols(0) {}
  void del(int c) {
    _data.col(c)=_data.col(_cols-1);
    _cols--;
  }
  void add(const Vec2T& pos) {
    if(_cols<_data.cols())
      _data.col(_cols++)=pos;
    else {
      Mat2XT tmp;
      tmp.resize(2,_data.cols()+1);
      tmp.block(0,0,2,_data.cols())=_data;
      tmp.col(_cols++)=pos;
      _data.swap(tmp);
    }
  }
  Mat2XTCM getCMap() const {
    return Mat2XTCM(_data.data(),2,_cols);
  }
  Mat2XTM getMap() {
    return Mat2XTM(_data.data(),2,_cols);
  }
  VecCM getCMapV() const {
    return VecCM(_data.data(),2*_cols);
  }
  VecM getMapV() {
    return VecM(_data.data(),2*_cols);
  }
  void reset() {
    _cols=0;
  }
  int rows() const {
    return 2;
  }
  int cols() const {
    return _cols;
  }
 private:
  Mat2XT _data;
  int _cols;
};
template <typename T>
struct DynamicVec {
  DECL_MAT_VEC_MAP_TYPES_T
  DECL_MAP_FUNCS
  DynamicVec():_rows(0) {}
  void del(int c) {
    _data[c]=_data[_rows-1];
    _rows--;
  }
  void add(T val) {
    if(_rows<_data.size())
      _data[_rows++]=val;
    else {
      Vec tmp;
      tmp.resize(_data.size()+1);
      tmp.segment(0,_data.size())=_data;
      tmp[_rows++]=val;
      _data.swap(tmp);
    }
  }
  VecCM getCMap() const {
    return VecCM(_data.data(),_rows);
  }
  VecM getMap() {
    return VecM(_data.data(),_rows);
  }
  void reset() {
    _rows=0;
  }
  int rows() const {
    return _rows;
  }
  int cols() const {
    return 1;
  }
 private:
  Vec _data;
  int _rows;
};
}

#endif

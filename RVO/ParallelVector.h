#ifndef PARALLEL_VECTOR_H
#define PARALLEL_VECTOR_H

#include <omp.h>
#include "Zero.h"
#include <Eigen/Sparse>

namespace RVO {
template <typename T>
class ParallelVector {
 public:
  typedef std::vector<T,Eigen::aligned_allocator<T> > vector_type;
  typedef typename std::vector<T,Eigen::aligned_allocator<T> >::const_iterator const_iterator;
  typedef typename std::vector<T,Eigen::aligned_allocator<T> >::iterator iterator;
  ParallelVector() {
    clear();
  }
  void clear() {
    clear(omp_get_num_procs());
  }
  void clear(int nr) {
    _blocks.assign(nr,vector_type());
  }
  void push_back(const T& newVal) {
    _blocks[id()].push_back(newVal);
  }
  template <typename IT>
  void insert(IT beg,IT end) {
    vector_type& v=_blocks[id()];
    v.insert(v.end(),beg,end);
  }
  const_iterator begin() const {
    const_cast<ParallelVector<T>&>(*this).join();
    return _blocks[0].begin();
  }
  const_iterator end() const {
    const_cast<ParallelVector<T>&>(*this).join();
    return _blocks[0].end();
  }
  iterator begin() {
    join();
    return _blocks[0].begin();
  }
  iterator end() {
    join();
    return _blocks[0].end();
  }
  const vector_type& getVector() const {
    const_cast<ParallelVector<T>*>(this)->join();
    return _blocks[0];
  }
  vector_type& getVector() {
    join();
    return _blocks[0];
  }
 protected:
  int id() const {
    return omp_get_thread_num()%(int)_blocks.size();
  }
  void join() {
    for(int i=1; i<(int)_blocks.size(); i++)
      if(!_blocks[i].empty()) {
        _blocks[0].insert(_blocks[0].end(),_blocks[i].begin(),_blocks[i].end());
        _blocks[i].clear();
      }
  }
  std::vector<vector_type> _blocks;
};
template <typename T>
class ParallelMatrix {
 public:
  typedef std::vector<T,Eigen::aligned_allocator<T> > vector_type;
  typedef typename std::vector<T,Eigen::aligned_allocator<T> >::const_iterator const_iterator;
  typedef typename std::vector<T,Eigen::aligned_allocator<T> >::iterator iterator;
  ParallelMatrix() {}
  ParallelMatrix(T example) {
    assign(omp_get_num_procs(),example);
  }
  ParallelMatrix(int nr,T example) {
    assign(nr,example);
  }
  void assign(T example) {
    assign(omp_get_num_procs(),example);
  }
  void assign(int nr,T example) {
    _blocks.assign(nr,example);
  }
  void clear() {
    _blocks[0]=Zero<T>::value(_blocks[0]);
    assign((int)_blocks.size(),_blocks[0]);
  }
  template <typename TOTHER>
  ParallelMatrix<T>& operator+=(const TOTHER& other) {
    _blocks[id()]+=other;
    return *this;
  }
  const T& getValueI() const {
    return _blocks[id()];
  }
  T& getValueI() {
    return _blocks[id()];
  }
  const T& getValue() const {
    const_cast<ParallelVector<T>*>(this)->join();
    return _blocks[0];
  }
  T& getValue() {
    join();
    return _blocks[0];
  }
 protected:
  int id() const {
    return omp_get_thread_num()%(int)_blocks.size();
  }
  void join() {
    for(int i=1; i<(int)_blocks.size(); i++) {
      _blocks[0]+=_blocks[i];
      _blocks[i]=Zero<T>::value(_blocks[0]);
    }
  }
  std::vector<T> _blocks;
};
template <typename T,int R,int C>
class ParallelMatrix<Eigen::Matrix<T,R,C>> {
 public:
  typedef Eigen::Matrix<T,R,C> Vec;
  typedef std::vector<Vec,Eigen::aligned_allocator<Vec> > vector_type;
  typedef typename std::vector<Vec,Eigen::aligned_allocator<Vec> >::const_iterator const_iterator;
  typedef typename std::vector<Vec,Eigen::aligned_allocator<Vec> >::iterator iterator;
  ParallelMatrix():_joined(true) {}
  ParallelMatrix(Vec example) {
    assign(omp_get_num_procs(),example);
  }
  ParallelMatrix(int nr,const Vec& example) {
    assign(nr,example);
  }
  void assign(const Vec& example) {
    assign(omp_get_num_procs(),example);
  }
  void assign(int nr,const Vec& example) {
    _blocks.assign(nr,example);
    _joined=true;
  }
  void clear() {
    _blocks[0]=Zero<Vec>::value(_blocks[0]);
    assign((int)_blocks.size(),_blocks[0]);
  }
  template <typename TOTHER>
  ParallelMatrix<T>& operator+=(const TOTHER& other) {
    _blocks[id()]+=other;
    _joined=false;
    return *this;
  }
  const Vec& getMatrixI() const {
    return _blocks[id()];
  }
  Vec& getMatrixI() {
    Vec& ret=_blocks[id()];
    _joined=false;
    return ret;
  }
  const Vec& getMatrix() const {
    const_cast<ParallelVector<T>*>(this)->join();
    return _blocks[0];
  }
  Vec& getMatrix() {
    join();
    return _blocks[0];
  }
 protected:
  int id() const {
    return omp_get_thread_num()%(int)_blocks.size();
  }
  void join() {
    if(_joined)
      return;
    for(int i=1; i<(int)_blocks.size(); i++) {
      _blocks[0]+=_blocks[i];
      _blocks[i]=Zero<Vec>::value(_blocks[0]);
    }
    _joined=true;
  }
  std::vector<Vec,Eigen::aligned_allocator<Vec>> _blocks;
  bool _joined;
};
}

#endif

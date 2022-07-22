#ifndef HEAP_H
#define HEAP_H

#include <functional>

namespace RVO {
template<class T>
struct absgreater : public std::binary_function<T, T, bool> {
  bool operator()(const T& left, const T& right) const {
    // apply operator> to operands
    return (std::abs(left) > std::abs(right));
  }
};
template<class T>
struct absless : public std::binary_function<T, T, bool> {
  bool operator()(const T& left, const T& right) const {
    // apply operator> to operands
    return (std::abs(left) < std::abs(right));
  }
};
//heap template on a compact vector where value and offset are separate
template <typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE,typename LESS>
typename HEAP_TYPE::value_type popHeapTpl(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& error_code) {
  typedef typename LESS::first_argument_type T;
  LESS le;

  if(heap.empty())
    return error_code;

  const typename HEAP_TYPE::value_type index=heap[0];
  heap[0]=heap.back();
  heapOffsets[heap[0]]=0;

  const int back=(int)(heap.size()-1);
  for(int i=0,j; i<back-1; i=j) {
    int lc=(i<<1)+1;
    int rc=lc+1;
    T current=result[heap[i]];
    T lv,rv;
    if(lc<back) {
      lv=result[heap[lc]];
      if(rc<back) {
        rv=result[heap[rc]];
        if(le(rv,lv)) {
          lc=rc;
          lv=rv;
        }
      }
      if(le(lv,current)) {
        heap[i]=heap[lc];
        heapOffsets[heap[lc]]=i;
        heap[lc]=heap.back();
        heapOffsets[heap.back()]=lc;
        j=lc;
      } else break;
    } else break;
  }

  heap.pop_back();
  return index;
}
template <typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE,typename LESS>
void pushHeapTpl(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& valueOff) {
  //typedef typename LESS::first_argument_type T;
  LESS le;

  const int back=(int)heap.size();
  heap.push_back(valueOff);
  heapOffsets[valueOff]=back;

  for(int i=back,j=(i-1)>>1; i>0; i=j,j=(i-1)>>1) {
    if(le(result[heap[i]],result[heap[j]])) {
      heap[i]=heap[j];
      heapOffsets[heap[j]]=i;
      heap[j]=valueOff;
      heapOffsets[valueOff]=j;
    } else break;
  }
}
template <typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE,typename LESS>
void updateHeapTpl(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& valueOff) {
  //typedef typename LESS::first_argument_type T;
  LESS le;

  for(int i=heapOffsets[valueOff],j=(i-1)>>1; i>0; i=j,j=(i-1)>>1) {
    if(le(result[heap[i]],result[heap[j]])) {
      heap[i]=heap[j];
      heapOffsets[heap[j]]=i;
      heap[j]=valueOff;
      heapOffsets[valueOff]=j;
    } else break;
  }
}
//heap min
template <typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE>
typename HEAP_TYPE::value_type popHeapDef(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& error_code) {
  typedef typename RESULT_TYPE::value_type T;
  return popHeapTpl<RESULT_TYPE,MARK_TYPE,HEAP_TYPE,std::less<T> >(result,heapOffsets,heap,error_code);
}
template <typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE>
void pushHeapDef(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& valueOff) {
  typedef typename RESULT_TYPE::value_type T;
  return pushHeapTpl<RESULT_TYPE,MARK_TYPE,HEAP_TYPE,std::less<T> >(result,heapOffsets,heap,valueOff);
}
template <typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE>
void updateHeapDef(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& valueOff) {
  typedef typename RESULT_TYPE::value_type T;
  return updateHeapTpl<RESULT_TYPE,MARK_TYPE,HEAP_TYPE,std::less<T> >(result,heapOffsets,heap,valueOff);
}
template <typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE>
typename HEAP_TYPE::value_type popHeapAbs(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& error_code) {
  typedef typename RESULT_TYPE::value_type T;
  return popHeapTpl<RESULT_TYPE,MARK_TYPE,HEAP_TYPE,absless<T> >(result,heapOffsets,heap,error_code);
}
template <typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE>
void pushHeapAbs(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& valueOff) {
  typedef typename RESULT_TYPE::value_type T;
  return pushHeapTpl<RESULT_TYPE,MARK_TYPE,HEAP_TYPE,absless<T> >(result,heapOffsets,heap,valueOff);
}
template <typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE>
void updateHeapAbs(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& valueOff) {
  typedef typename RESULT_TYPE::value_type T;
  return updateHeapTpl<RESULT_TYPE,MARK_TYPE,HEAP_TYPE,absless<T> >(result,heapOffsets,heap,valueOff);
}
//heap max
template<typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE>
typename HEAP_TYPE::value_type popHeapMaxDef(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& error_code) {
  typedef typename RESULT_TYPE::value_type T;
  return popHeapTpl<RESULT_TYPE,MARK_TYPE,HEAP_TYPE,std::greater<T> >(result,heapOffsets,heap,error_code);
}
template<typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE>
void pushHeapMaxDef(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& valueOff) {
  typedef typename RESULT_TYPE::value_type T;
  return pushHeapTpl<RESULT_TYPE,MARK_TYPE,HEAP_TYPE,std::greater<T> >(result,heapOffsets,heap,valueOff);
}
template<typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE>
void updateHeapMaxDef(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& valueOff) {
  typedef typename RESULT_TYPE::value_type T;
  return updateHeapTpl<RESULT_TYPE,MARK_TYPE,HEAP_TYPE,std::greater<T> >(result,heapOffsets,heap,valueOff);
}
template<typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE>
typename HEAP_TYPE::value_type popHeapMaxAbs(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& error_code) {
  typedef typename RESULT_TYPE::value_type T;
  return popHeapTpl<RESULT_TYPE,MARK_TYPE,HEAP_TYPE,absgreater<T> >(result,heapOffsets,heap,error_code);
}
template<typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE>
void pushHeapMaxAbs(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& valueOff) {
  typedef typename RESULT_TYPE::value_type T;
  return pushHeapTpl<RESULT_TYPE,MARK_TYPE,HEAP_TYPE,absgreater<T> >(result,heapOffsets,heap,valueOff);
}
template<typename RESULT_TYPE,typename MARK_TYPE,typename HEAP_TYPE>
void updateHeapMaxAbs(const RESULT_TYPE& result,MARK_TYPE& heapOffsets,HEAP_TYPE& heap,const typename HEAP_TYPE::value_type& valueOff) {
  typedef typename RESULT_TYPE::value_type T;
  return updateHeapTpl<RESULT_TYPE,MARK_TYPE,HEAP_TYPE,absgreater<T> >(result,heapOffsets,heap,valueOff);
}
}

#endif

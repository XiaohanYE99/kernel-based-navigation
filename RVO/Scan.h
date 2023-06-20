#ifndef SCAN_H
#define SCAN_H

#include <omp.h>
#include <vector>

namespace RVO {
template <typename T,typename Op>
void omp_scan(int n,const T* in,T* out,Op op) {
  int i,last_value_chunk_array[1000],chunk;
  //parallel region begins
  #pragma omp parallel shared(in,out,chunk) private(i)
  {
    const int num_threads=omp_get_num_threads();    //get #threads
    chunk=(n+num_threads-1)/num_threads;            //#elements per thread
    const int idthread=omp_get_thread_num();        //ID of thread
    #pragma omp single
    {
      last_value_chunk_array[0]=0;
    }
    int operation = 0;
    //For region begins
    #pragma omp for schedule(static,chunk) nowait
    for(i=0; i<n; i++) {
      if((i%chunk)==0) {
        operation=in[i];                            //breaking at every chunk
        out[i]=in[i];
      } else {
        out[i]=op(out[i-1],in[i]);                  //performing the required operation
        operation=op(operation,in[i]);
      }
    }
    //For region ends
    last_value_chunk_array[idthread+1]=operation;   //assigning sums of all chunks in last_chunk_value array

    #pragma omp barrier                             //syncing all the threads
    int balance=last_value_chunk_array[1];          //initialising with index 1 value as for thread 0, result has already been calculated
    if(idthread==1)
      balance=last_value_chunk_array[1];            //for thread ID==1

    for(int i=2; i<idthread+1; i++)
      balance=op(balance,last_value_chunk_array[i]);//creating balance for every thread

    #pragma omp for schedule(static,chunk)          //to calculate the sum of all chunks
    for(int i=0; i<n; i++)
      if(idthread!=0)
        out[i]=op(out[i],balance);                  //for thread IDs other than 0
  }
  //parallel region ends
}
template <typename T>
void omp_scan_add(int n,const T* in,T* out) {
  return omp_scan(n,in,out,[&](T a,T b) {
    return a+b;
  });
}
template <typename T>
void omp_scan_add(const std::vector<T>& in,std::vector<T>& out) {
  out.resize(in.size());
  return omp_scan(in.size(),in.data(),out.data(),[&](T a,T b) {
    return a+b;
  });
}
}

#endif

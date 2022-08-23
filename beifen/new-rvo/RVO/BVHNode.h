#ifndef BVH_NODE_H
#define BVH_NODE_H

#include "BBox.h"
#include "Obstacle.h"
#include <unordered_set>
#include <vector>

namespace RVO {
template <typename T>
struct EdgeHash {
  static size_t hashCombine(size_t h1,size_t h2) {
    return h1^(h2<<1);
  }
  size_t operator()(const Eigen::Matrix<T,2,1>& key) const {
    std::hash<T> h;
    return hashCombine(h(key[0]),h(key[1]));
  }
  bool operator()(const Eigen::Matrix<T,2,1>& a,const Eigen::Matrix<T,2,1>& b) const {
    for(int i=0; i<2; i++)
      if(a[i]<b[i])
        return true;
      else if(a[i]>b[i])
        return false;
    return false;
  }
};
template <typename T,typename BBOX>
struct Node {
  typedef BBOX BoxType;
  Node();
  Node<T,BBOX>& operator=(const Node<T,BBOX>& other);
  static void update(std::vector<Node<T,BBOX>>& bvh,std::function<BBOX(const T& n)> getBB);
  static void update(int i,std::vector<Node<T,BBOX>>& bvh,std::function<BBOX(const T& n)> getBB);
  static void buildBVHBottomUp(std::vector<Node<T,BBOX>>& bvh,const std::unordered_set<Eigen::Matrix<int,2,1>,EdgeHash<int>>& edgeMap,bool singleComponent=false);
  int _l,_r,_parent,_nrCell;
  BBOX _bb;
  T _cell;
};
}

#endif

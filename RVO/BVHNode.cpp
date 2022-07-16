#include "BVHNode.h"
#include "Heap.h"

namespace RVO {
//SurfaceArea
template <int D>
struct SurfaceArea;
template <>
struct SurfaceArea<2> {
  template <typename BoxType>
  static typename BoxType::Vec2T::Scalar area(const BoxType& bb) {
    typedef typename BoxType::Vec2T Vec2T;
    Vec2T ext=Vec2T::Zero().cwiseMax(bb.maxCorner()-bb.minCorner());
    return (ext[0]+ext[1])*2.0f;
  }
};
//Update field _fixed
template <typename T,typename BBOX>
struct NodeTraits {
  static bool isLeaf(Node<T,BBOX>& n) {
    return n._cell!=NULL;
  }
  static T empty() {
    return NULL;
  }
};
template <typename BBOX>
struct NodeTraits<int,BBOX> {
  static bool isLeaf(Node<int,BBOX>& n) {
    return n._cell>=0;
  }
  static int empty() {
    return -1;
  }
};
//Node
template <typename T,typename BBOX>
Node<T,BBOX>::Node():_l(-1),_r(-1),_parent(-1),_nrCell(-1) {}
template <typename T,typename BBOX>
Node<T,BBOX>& Node<T,BBOX>::operator=(const Node<T,BBOX>& other) {
  _bb=other._bb;
  _cell=other._cell;
  _l=other._l;
  _r=other._r;
  _parent=other._parent;
  _nrCell=other._nrCell;
  return *this;
}
template <typename T,typename BBOX>
void Node<T,BBOX>::update(std::vector<Node<T,BBOX>>& bvh,std::function<BBOX(const T& n)> getBB) {
  if(bvh.empty())
    return;
  update((int)bvh.size()-1,bvh,getBB);
}
template <typename T,typename BBOX>
void Node<T,BBOX>::update(int i,std::vector<Node<T,BBOX>>& bvh,std::function<BBOX(const T& n)> getBB) {
  auto& n=bvh[i];
  if(n._cell) {
    n._bb=getBB(n._cell);
  } else  {
    update(n._l,bvh,getBB);
    update(n._r,bvh,getBB);
    n._bb=bvh[n._l]._bb;
    n._bb.setUnion(bvh[n._r]._bb);
  }
}
template <typename T,typename BBOX>
void Node<T,BBOX>::buildBVHBottomUp(std::vector<Node<T,BBOX>>& bvh,const std::unordered_set<Eigen::Matrix<int,2,1>,EdgeHash<int>>& edgeMap,bool singleComponent) {
  //initialize hash
  std::vector<int> heap;
  std::vector<int> heapOffsets;
  std::vector<Eigen::Matrix<int,2,1>> ess;
  std::vector<typename BBOX::Vec2T::Scalar> cost;
  for(auto beg=edgeMap.begin(),end=edgeMap.end(); beg!=end; beg++) {
    heapOffsets.push_back(-1);
    ess.push_back(*beg);
    BBOX bb=bvh[beg->coeff(0)]._bb;
    if(beg->coeff(1)>=0)
      bb.setUnion(bvh[beg->coeff(1)]._bb);
    typename BBOX::Vec2T::Scalar c=SurfaceArea<2>::area(bb);
    cost.push_back(c);
  }
  for(int i=0; i<(int)ess.size(); i++)
    pushHeapDef(cost,heapOffsets,heap,i);
  //merge BVH
  int err=-1;
  while(!heap.empty()) {
    int i=popHeapDef(cost,heapOffsets,heap,err);
    int t0=ess[i][0],t1=ess[i][1];
    //boundary edge
    if(t1==-1)
      continue;
    //find parent
    while(bvh[t0]._parent>=0)
      t0=bvh[t0]._parent;
    while(bvh[t1]._parent>=0)
      t1=bvh[t1]._parent;
    //check already merged
    if(t0==t1)
      continue;
    //merge
    BBOX bb=bvh[t0]._bb;
    bb.setUnion(bvh[t1]._bb);
    typename BBOX::Vec2T::Scalar c=SurfaceArea<2>::area(bb);
    if(c>cost[i]) {
      cost[i]=c;
      pushHeapDef(cost,heapOffsets,heap,i);
    } else {
      Node<T,BBOX> n;
      n._l=t0;
      n._r=t1;
      n._parent=-1;
      n._cell=NodeTraits<T,BBOX>::empty();
      n._bb=bb;
      n._nrCell=bvh[n._l]._nrCell+bvh[n._r]._nrCell;
      bvh[t0]._parent=(int)bvh.size();
      bvh[t1]._parent=(int)bvh.size();
      bvh.push_back(n);
    }
  }
  //check or merge root
  if(singleComponent) {
    int nrLeaf=0;
    for(auto n:bvh)
      if(NodeTraits<T,BBOX>::isLeaf(n))
        nrLeaf++;
    ASSERT_MSG((int)bvh.size()==nrLeaf*2-1,"Multi-component mesh detected!")
  } else {
    std::vector<int> roots;
    for(int i=0; i<(int)bvh.size(); i++)
      if(bvh[i]._parent==-1)
        roots.push_back(i);
    //build edge map for roots
    std::unordered_set<Eigen::Matrix<int,2,1>,EdgeHash<int>> edgeMapRoot;
    for(int i=0; i<(int)roots.size(); i++)
      for(int j=i+1; j<(int)roots.size(); j++)
        edgeMapRoot.insert(Eigen::Matrix<int,2,1>(roots[i],roots[j]));
    //merge root
    buildBVHBottomUp(bvh,edgeMapRoot,true);
  }
}
template struct Node<std::shared_ptr<Obstacle>,BBox>;
}

#ifndef VISIBILITY_H
#define VISIBILITY_H

#include "RVO.h"

namespace RVO {
struct PolarInterval {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  PolarInterval();
  PolarInterval(const Vec2T& dL,const Vec2T& dR,int idL=-1,int idR=-1);
  bool withinAngle(const Vec2T& d) const;
  bool within(const Vec2T& d,T& alpha) const;
  bool valid() const;
  bool wrapAround() const;
  void divide(PolarInterval& L,PolarInterval& R) const;
  //data
  Vec2T _dL,_dR;
  int _idL,_idR;
};
struct PolarIntervals {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  bool less(const std::pair<int,bool>& a,const std::pair<int,bool>& b) const;
  const PolarInterval& interval(const std::pair<int,bool>& pss) const;
  const Vec2T& dir(const std::pair<int,bool>& p) const;
  bool isNegX(const std::pair<int,bool>& p) const;
  T angle(const std::pair<int,bool>& p) const;
  int id(const std::pair<int,bool>& p) const;
  //visibility
  void visible(std::unordered_set<int>& pss,std::function<bool(int,const Vec2T&)> canAdd);
  void addInterval(const PolarInterval& I);
  void updateHeap(int ptr);
  void sort();
  //data
  std::vector<T> _distance;
  std::vector<int> _heapOffset,_heap;
  std::vector<PolarInterval> _intervals;
  std::vector<std::pair<int,bool>> _pointers; //<id,left>
};
struct ShortestPath {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  Vec2T _target;
  std::vector<int> _last;
  std::vector<T> _distance;
  T _maxVelocity;
  Mat2T _DVDP;
};
class VisibilityGraph {
 public:
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  enum Label {
    OUT_OF_REACH=-1,
    TARGET=-2,
  };
  VisibilityGraph(RVOSimulator& rvo);
  VisibilityGraph(RVOSimulator& rvo,const VisibilityGraph& other);
  virtual ~VisibilityGraph();
  std::vector<std::pair<Vec2T,Vec2T>> lines(const Vec2T& p) const;
  std::vector<std::pair<Vec2T,Vec2T>> lines(int id=-1) const;
  void findNeighbor(int id,int& idNext,int& idLast) const;
  std::unordered_set<int> visible(const Vec2T& p,int id=-1) const;
  ShortestPath buildShortestPath(const Vec2T& target) const;
  void setAgentTarget(int i,const Vec2T& target,T maxVelocity);
  int getNrBoundaryPoint() const;
  Vec2T getAgentWayPoint(const ShortestPath& p,const Vec2T& pos) const;
  Vec2T getAgentWayPoint(int i) const;
  Mat2T getAgentDVDP(int i) const;
  void updateAgentTargets();
 protected:
  RVOSimulator& _rvo;
  std::vector<std::unordered_set<int>> _graph;
  std::unordered_map<int,ShortestPath> _paths;
};
}

#endif

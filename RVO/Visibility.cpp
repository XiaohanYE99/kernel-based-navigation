#include "Visibility.h"
#include "Heap.h"

namespace RVO {
template <typename T>
T cross(const Eigen::Matrix<T,2,1>& a,const Eigen::Matrix<T,2,1>& b) {
  return a[0]*b[1]-a[1]*b[0];
}
//PolarInterval
PolarInterval::PolarInterval() {}
PolarInterval::PolarInterval(const Vec2T& dL,const Vec2T& dR,int idL,int idR):_dL(dL),_dR(dR),_idL(idL),_idR(idR) {}
bool PolarInterval::withinAngle(const Vec2T& d) const {
  return cross(_dL,d)>0 && cross(d,_dR)>0;
}
bool PolarInterval::within(const Vec2T& d,T& alpha) const {
  Mat2T LHS;
  Vec2T RHS=_dL;
  LHS.col(0)=d;
  LHS.col(1)=_dL-_dR;
  alpha=(LHS.inverse()*RHS)[0];
  return withinAngle(d) && alpha<1;
}
bool PolarInterval::valid() const {
  return !_dL.isZero() && !_dR.isZero() && cross(_dL,_dR)>0;
}
bool PolarInterval::wrapAround() const {
  return _dL.y()>0 && _dR.y()<0;
}
void PolarInterval::divide(PolarInterval& L,PolarInterval& R) const {
  ASSERT_MSG(wrapAround(),"Cannot divide interval that does not warp around!")
  T alpha=0;
  within(Vec2T(-1,0),alpha);
  L=PolarInterval(_dL,Vec2T(-alpha,0),_idL,-1);
  R=PolarInterval(Vec2T(-alpha,0),_dR,-1,_idR);
  ASSERT_MSG(L.valid() && R.valid(),"Invalid interval division!")
}
//PolarIntervals
bool PolarIntervals::less(const std::pair<int,bool>& a,const std::pair<int,bool>& b) const {
  //angle
  T aA=angle(a),aB=angle(b);
  if(aA<aB)
    return true;
  else if(aA>aB)
    return false;
  //left/right
  if(a.second && !b.second)
    return true;
  else if(!a.second && b.second)
    return false;
  return false;
}
const PolarInterval& PolarIntervals::interval(const std::pair<int,bool>& p) const {
  return _intervals[p.first];
}
const PolarIntervals::Vec2T& PolarIntervals::dir(const std::pair<int,bool>& p) const {
  return p.second?interval(p)._dL:interval(p)._dR;
}
bool PolarIntervals::isNegX(const std::pair<int,bool>& p) const {
  const Vec2T& d=dir(p);
  return d[1]==0 && d[0]<0;
}
PolarIntervals::T PolarIntervals::angle(const std::pair<int,bool>& p) const {
  if(isNegX(p))
    return p.second?-M_PI:M_PI;
  else return atan2(dir(p)[1],dir(p)[0]);
}
int PolarIntervals::id(const std::pair<int,bool>& p) const {
  return p.second?interval(p)._idL:interval(p)._idR;
}
//visibility
void PolarIntervals::visible(std::unordered_set<int>& pss,std::function<bool(int,const Vec2T&)> canAdd) {
//#define BRUTE_FORCE
#ifdef BRUTE_FORCE
  for(int ptr=0; ptr<(int)_pointers.size(); ptr++) {
    const Vec2T& d=dir(_pointers[ptr]);
    int did=id(_pointers[ptr]);
    bool valid=true;
    T alpha;
    for(int i=0; valid && i<(int)_intervals.size(); i++) {
      if(i==_pointers[ptr].first)
        continue;
      if(!_intervals[i].within(d,alpha))
        continue;
      if(alpha<1)
        valid=false;
    }
    if(valid)
      if(did>=0 && canAdd(did,d))
        pss.insert(did);
  }
#else
  sort();
  //since we break intervals along -x axis, this axis is a special case
  T distNegX=std::numeric_limits<double>::max();
  for(int ptr=0; ptr<(int)_pointers.size(); ptr++)
    if(isNegX(_pointers[ptr]))
      distNegX=std::min(distNegX,-dir(_pointers[ptr])[0]);
  //main loop
  std::vector<int> heapTmp;
  for(int ptr=0; ptr<(int)_pointers.size(); ptr++) {
    const Vec2T& d=dir(_pointers[ptr]);
    int did=id(_pointers[ptr]);
    //update heap value
    heapTmp=_heap;
    for(int Iid:heapTmp) {
      _intervals[Iid].within(d,_distance[Iid]);
      updateHeapDef(_distance,_heapOffset,_heap,Iid);
    }
    //move pointer forward
    updateHeap(ptr);
    //since we break intervals along -x axis, this axis is a special case
    if(isNegX(_pointers[ptr]))
      if(-d[0]>distNegX)
        continue;
    //insert if visible
    if(_heap.empty() || _distance[_heap[0]]>=1)
      if(did>=0 && canAdd(did,d))
        pss.insert(did);
  }
#endif
}
void PolarIntervals::addInterval(const PolarInterval& I) {
  if(I.wrapAround()) {
    PolarInterval L,R;
    I.divide(L,R);
    addInterval(L);
    addInterval(R);
  } else {
    _pointers.push_back(std::make_pair((int)_intervals.size(),true));
    _pointers.push_back(std::make_pair((int)_intervals.size(),false));
    _intervals.push_back(I);
  }
}
void PolarIntervals::updateHeap(int ptr) {
  std::vector<int>::value_type err;
  if(_pointers[ptr].second) {
    //insert interval into heap
    _distance[_pointers[ptr].first]=1;
    pushHeapDef(_distance,_heapOffset,_heap,_pointers[ptr].first);
  } else {
    //remove interval from heap
    _distance[_pointers[ptr].first]=std::numeric_limits<double>::min();
    updateHeapDef(_distance,_heapOffset,_heap,_pointers[ptr].first);
    popHeapDef(_distance,_heapOffset,_heap,err);
  }
}
void PolarIntervals::sort() {
  _distance.assign((int)_intervals.size(),0);
  _heapOffset.assign((int)_intervals.size(),-1);
  _heap.clear();
  std::sort(_pointers.begin(),_pointers.end(),[&](const std::pair<int,bool>& a,const std::pair<int,bool>& b) {
    return less(a,b);
  });
}
//Visibility
VisibilityGraph::VisibilityGraph(RVOSimulator& rvo):_rvo(rvo) {
  const auto& obs=_rvo.getBVH().getObstacles();
  _graph.resize((int)obs.size());
  OMP_PARALLEL_FOR_
  for(int i=0; i<(int)obs.size(); i++)
    _graph[i]=visible(obs[i]->_pos,i);
}
VisibilityGraph::VisibilityGraph(RVOSimulator& rvo,const VisibilityGraph& other):_rvo(rvo) {
  _graph=other._graph;
}
std::vector<std::pair<VisibilityGraph::Vec2T,VisibilityGraph::Vec2T>> VisibilityGraph::lines(const Vec2T& p) const {
  const auto& obs=_rvo.getBVH().getObstacles();
  std::vector<std::pair<Vec2T,Vec2T>> lines;
  for(int id:visible(p))
    lines.push_back(std::make_pair(p,obs[id]->_pos));
  return lines;
}
std::vector<std::pair<VisibilityGraph::Vec2T,VisibilityGraph::Vec2T>> VisibilityGraph::lines(int id) const {
  const auto& obs=_rvo.getBVH().getObstacles();
  std::vector<std::pair<Vec2T,Vec2T>> lines;
  for(int i=0; i<(int)_graph.size(); i++)
    if(id<0 || i==id)
      for(int j:_graph[i])
        lines.push_back(std::make_pair(obs[i]->_pos,obs[j]->_pos));
  return lines;
}
void VisibilityGraph::findNeighbor(int id,int& idNext,int& idLast) const {
  const auto& obs=_rvo.getBVH().getObstacles();
  idLast=idNext=obs[id]->_next->_id;
  while(obs[idLast]->_next->_id!=id)
    idLast=obs[idLast]->_next->_id;
}
std::unordered_set<int> VisibilityGraph::visible(const Vec2T& p,int id) const {
  const auto& obs=_rvo.getBVH().getObstacles();
  std::vector<bool> visited(obs.size(),false);
  std::unordered_set<int> pss;
  //this point is part of boundary
  PolarInterval incidentI;
  int idLast,idNext;
  if(id>=0) {
    findNeighbor(id,idNext,idLast);
    incidentI=PolarInterval(obs[idNext]->_pos-p,obs[idLast]->_pos-p);
    //do not calculate visibility for concave vertex
    if(!incidentI.valid())
      return pss;
    else {
      if(_rvo.getBVH().visible(obs[id]->_pos,obs[idLast]->_pos,obs[id]))
        pss.insert(idLast);
      if(_rvo.getBVH().visible(obs[id]->_pos,obs[idNext]->_pos,obs[id]))
        pss.insert(idNext);
    }
  }
  //insert intervals
  PolarIntervals intervals;
  for(int i=0; i<(int)visited.size(); i++) {
    if(visited[i])
      continue;
    std::shared_ptr<Obstacle> curr=obs[i];
    while(!visited[curr->_id]) {
      PolarInterval I(curr->_next->_pos-p,curr->_pos-p,curr->_next->_id,curr->_id);
      if(I.valid())
        intervals.addInterval(I);
      visited[curr->_id]=true;
      curr=curr->_next;
    }
  }
  //compute visibility
  intervals.visible(pss,[&](int pid,const Vec2T& dir) {
    if(id<0)
      return true;
    return pid!=id && pid!=idLast && pid!=idNext && !incidentI.withinAngle(dir);
  });
  return pss;
}
VisibilityGraph::ShortestPath VisibilityGraph::buildShortestPath(const Vec2T& target) {
  const auto& obs=_rvo.getBVH().getObstacles();
  ShortestPath path;
  path._target=target;
  path._last.assign(obs.size(),OUT_OF_REACH);
  path._distance.assign(obs.size(),std::numeric_limits<double>::max());
  //Dijkstra
  std::vector<int> heap;
  std::vector<int> heapOffsets(obs.size(),-1);
  //initialize
  for(int id:visible(target)) {
    path._last[id]=TARGET;
    path._distance[id]=(target-obs[id]->_pos).norm();
  }
  for(int i=0; i<(int)obs.size(); i++)
    pushHeapDef(path._distance,heapOffsets,heap,i);
  //main loop
  int err;
  while(!heap.empty()) {
    int i=popHeapDef(path._distance,heapOffsets,heap,err);
    for(int other:_graph[i]) {
      T alt=path._distance[i]+(obs[other]->_pos-obs[i]->_pos).norm();
      if(alt<path._distance[other]) {
        path._distance[other]=alt;
        path._last[other]=i;
        updateHeapDef(path._distance,heapOffsets,heap,other);
      }
    }
  }
  return path;
}
void VisibilityGraph::setAgentTarget(int i,const Vec2T& target,T maxVelocity) {
  _paths[i]=buildShortestPath(target);
  _paths[i]._maxVelocity=maxVelocity;
}
int VisibilityGraph::getNrBoundaryPoint() const {
  return (int)_graph.size();
}
VisibilityGraph::Vec2T VisibilityGraph::getWayPoint(int i) const {
  const auto& obs=_rvo.getBVH().getObstacles();
  const ShortestPath& p=_paths.find(i)->second;
  Vec2T pos=_rvo.getAgentPosition(i);
  if(_rvo.getBVH().visible(pos,p._target))
    return p._target;
  else {
    int minId=-1;
    T minDistance=std::numeric_limits<double>::max();
    for(int id:visible(_rvo.getAgentPosition(i))) {
      T distance=(pos-obs[id]->_pos).norm()+p._distance[id];
      if(distance<minDistance) {
        minDistance=distance;
        minId=id;
      }
    }
    ASSERT_MSG(minId>=0,"Target out of reach!")
    return obs[minId]->_pos;
  }
}
void VisibilityGraph::updateAgentTargets() {
  for(const auto& p:_paths) {
    int i=p.first;
    Vec2T pos=_rvo.getAgentPosition(i),dir=getWayPoint(i)-pos;
    T len=dir.norm();
    if(len>p.second._maxVelocity)
      _rvo.setAgentVelocity(i,dir*p.second._maxVelocity/len);
    else _rvo.setAgentVelocity(i,dir);
  }
}
}

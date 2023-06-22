#include "BoundingVolumeHierarchy.h"
#include <boost/polygon/polygon.hpp>
#include <stack>

namespace RVO {
BoundingVolumeHierarchy::BoundingVolumeHierarchy() {}
BoundingVolumeHierarchy::BoundingVolumeHierarchy(const BoundingVolumeHierarchy& other,bool simplify):_obs(other._obs) {
  //copy obs
  _obs.resize(other._obs.size());
  for(int i=0; i<(int)_obs.size(); i++)
    _obs[i].reset(new Obstacle(other._obs[i]->_pos,other._obs[i]->_id));
  for(int i=0; i<(int)_obs.size(); i++)
    _obs[i]->_next=_obs[other._obs[i]->_next->_id];
  //assemble
  if(simplify)
    assembleSimplified();
  else assembleFull();
}
BoundingVolumeHierarchy::~BoundingVolumeHierarchy() {
  clearObstacle();
}
void BoundingVolumeHierarchy::clearObstacle() {
  for(auto& obs:_obs)
    obs->_next=NULL;
  _backup.insert(_backup.end(),_obs.begin(),_obs.end());
  _obs.clear();
  _bvh.clear();
}
int BoundingVolumeHierarchy::getNrObstacle() const {
  int nr=0;
  for(int id=0; id<(int)_obs.size(); id++)
    if(_obs[id]->_next->_id<id)
      nr++;
  return nr;
}
std::vector<BoundingVolumeHierarchy::Vec2T> BoundingVolumeHierarchy::getObstacle(int i) const {
  int nr=0;
  std::vector<Vec2T> pos;
  for(int id=0; id<(int)_obs.size(); id++)
    if(_obs[id]->_next->_id<id) {
      if(nr==i)
        for(int idBeg=_obs[id]->_next->_id; idBeg<=id; idBeg++)
          pos.push_back(_obs[idBeg]->_pos);
      nr++;
    }
  return pos;
}
int BoundingVolumeHierarchy::getNrVertex() const {
  return (int)_obs.size();
}
std::shared_ptr<Obstacle> BoundingVolumeHierarchy::getVertex(int i) const {
  return _obs[i];
}
void BoundingVolumeHierarchy::addObstacle(const std::vector<Vec2T>& vss) {
  addObstacleInternal(vss);
  assembleFull();
}
const std::vector<std::shared_ptr<Obstacle>>& BoundingVolumeHierarchy::getObstacles() const {
  return _obs;
}
const std::vector<Node<int,BBox>>& BoundingVolumeHierarchy::getNodes() const {
  return _bvh;
}
bool BoundingVolumeHierarchy::visible(const Vec2T& a,const Vec2T& b,std::shared_ptr<Obstacle> obs) const {
  BBox bb;
  bb.setUnion(a);
  bb.setUnion(b);
  Vec2T edgeA[2]= {a,b};
  //stack
  std::stack<int> ss;
  ss.push((int)_bvh.size()-1);
  while(!ss.empty()) {
    int curr=ss.top();
    ss.pop();
    if(!_bvh[curr]._bb.intersect(bb))
      continue;
    else if(_bvh[curr]._cell>=0) {
      if(obs) {
        if(_obs[_bvh[curr]._cell]==obs || _obs[_bvh[curr]._cell]->_next==obs)
          continue;
        if(_obs[_bvh[curr]._cell]==obs->_next || _obs[_bvh[curr]._cell]->_next==obs->_next)
          continue;
      }
      Vec2T edgeB[2]= {_obs[_bvh[curr]._cell]->_pos,_obs[_bvh[curr]._cell]->_next->_pos};
      if(intersect(edgeA,edgeB))
        return false;
    } else {
      ss.push(_bvh[curr]._l);
      ss.push(_bvh[curr]._r);
    }
  }
  return true;
}
bool BoundingVolumeHierarchy::visible(const Vec2T& a,std::shared_ptr<Obstacle> obs,Vec2T* bRef) const {
  Vec2T o[2]= {obs->_pos,obs->_next->_pos};
  Vec2T obsVec=o[1]-o[0],relPos0=o[0]-a;
  T lenSq=obsVec.squaredNorm(),s=(-relPos0.dot(obsVec))/lenSq;
  s=fmin((T)1.,fmax((T)0.,s));
  Vec2T b=o[0]*(1-s)+o[1]*s;
  if(bRef)
    *bRef=b;
  return visible(a,b,obs);
}
bool BoundingVolumeHierarchy::intersect(const Vec2T edgeA[2],const Vec2T edgeB[2]) {
  //edgeA[0]+s*(edgeA[1]-edgeA[0])=edgeB[0]+t*(edgeB[1]-edgeB[0])
  Mat2T LHS;
  Vec2T RHS=edgeB[0]-edgeA[0];
  LHS.col(0)= (edgeA[1]-edgeA[0]);
  LHS.col(1)=-(edgeB[1]-edgeB[0]);
  if(fabs(LHS.determinant())<Epsilon<T>::defaultEps()) {
    return false;   //parallel line segment, doesn't matter
  } else {
    Vec2T st=LHS.inverse()*RHS;
    return st[0]>=0 && st[0]<=1 && st[1]>=0 && st[1]<=1;
  }
}
BoundingVolumeHierarchy::T BoundingVolumeHierarchy::distanceAgentObstacle(const Vec2T edgeA[2],const Vec2T edgeB[2]) {
  if(intersect(edgeA,edgeB))
    return 0;
  else {
    T d0=distance(edgeA[0],edgeB);
    T d1=distance(edgeA[1],edgeB);
    T d2=distance(edgeB[0],edgeA);
    T d3=distance(edgeB[1],edgeA);
    return fmin(fmin(d0,d1),fmin(d2,d3));
  }
}
BoundingVolumeHierarchy::T BoundingVolumeHierarchy::distanceAgentAgent(const Vec2T edgeA[2],const Vec2T edgeB[2]) {
  //f=|A+B*t|^2=a*t^2+b*t+c
  Vec2T A=edgeA[0]-edgeB[0],B=(edgeA[1]-edgeA[0])-(edgeB[1]-edgeB[0]);
  T a=B.squaredNorm(),b=A.dot(B)*2,c=A.squaredNorm();
  T minT=-b/2/a;
  if(minT>=0 && minT<=1)
    return sqrt(a*minT*minT+b*minT+c);
  else if(minT<0)
    return sqrt(c);
  else return sqrt(a+b+c);
}
BoundingVolumeHierarchy::T BoundingVolumeHierarchy::distance(const Vec2T& pt,const Vec2T edgeB[2]) {
  //f=|A+B*t|^2=a*t^2+b*t+c
  Vec2T A=edgeB[0]-pt,B=edgeB[1]-edgeB[0];
  T a=B.squaredNorm(),b=A.dot(B)*2,c=A.squaredNorm();
  T minT=-b/2/a;
  if(minT>=0 && minT<=1)
    return sqrt(a*minT*minT+b*minT+c);
  else if(minT<0)
    return sqrt(c);
  else return sqrt(a+b+c);
}
BoundingVolumeHierarchy::T BoundingVolumeHierarchy::closestT(const Vec2T& pt,const Vec2T edgeB[2]) {
  //f=|A+B*t|^2=a*t^2+b*t+c
  Vec2T A=edgeB[0]-pt,B=edgeB[1]-edgeB[0];
  T a=B.squaredNorm(),b=A.dot(B)*2;
  return fmax((T)0,fmin((T)1,-b/2/a));
}
//helper
void BoundingVolumeHierarchy::assembleFull() {
  _bvh.clear();
  std::unordered_set<Vec2i,EdgeHash<int>> edgeMap;
  for(int i=0; i<(int)_obs.size(); i++) {
    _bvh.push_back(Node<int,BBox>());
    _bvh.back()._nrCell=1;
    _bvh.back()._cell=i;
    _bvh.back()._bb.setUnion(_obs[i]->_pos);
    _bvh.back()._bb.setUnion(_obs[i]->_next->_pos);
    //consecutive edges
    if(i<(int)_obs.size()-1 && _obs[i]->_next==_obs[i+1])
      edgeMap.insert(Vec2i(i,i+1));
  }
  Node<int,BBox>::buildBVHBottomUp(_bvh,edgeMap,false);
}
void BoundingVolumeHierarchy::assembleSimplified() {
  namespace gtl=boost::polygon;
  using namespace boost::polygon::operators;
  typedef long long int T2;
  typedef gtl::polygon_with_holes_data<T2> Polygon;
  typedef gtl::polygon_traits<Polygon>::point_type Point;
  typedef std::vector<Polygon> PolygonSet;
  //insert and union polygons
  PolygonSet ps;
  std::vector<bool> visited(_obs.size(),false);
  for(int i=0; i<(int)visited.size(); i++) {
    if(visited[i])
      continue;
    std::vector<Point> pss;
    std::shared_ptr<Obstacle> curr=_obs[i];
    while(!visited[curr->_id]) {
      for(int d=0; d<2; d++)
        if(curr->_pos[d]!=T2(curr->_pos[d]))
          throw std::invalid_argument("Point coordinate is not integer!");
      pss.push_back(gtl::construct<Point>(curr->_pos[0],curr->_pos[1]));
      visited[curr->_id]=true;
      curr=curr->_next;
    }
    Polygon poly;
    gtl::set_points(poly,pss.begin(),pss.end());
    ps+=poly;
  }
  //insert to bvh
  _obs.clear();
  for(const auto& p:ps) {
    //outer
    {
      std::vector<Eigen::Matrix<T2,2,1>> vss;
      for(const auto& v:p)
        vss.push_back(Eigen::Matrix<T2,2,1>(v.x(),v.y()));
      removeDuplicateVertices(vss);
      addObstacleInternal(vss);
    }
    //hole
    for(const auto& h:p.holes_) {
      std::vector<Eigen::Matrix<T2,2,1>> vss;
      for(const auto& v:h)
        vss.push_back(Eigen::Matrix<T2,2,1>(v.x(),v.y()));
      removeDuplicateVertices<T2>(vss);
      addObstacleInternal<T2>(vss);
    }
  }
  assembleFull();
}
template <typename T2>
void BoundingVolumeHierarchy::addObstacleInternal(const std::vector<Eigen::Matrix<T2,2,1>>& vss) {
  int offset=(int)_obs.size();
  for(int i=0; i<(int)vss.size(); i++)
    if(_backup.empty())
      _obs.push_back(std::shared_ptr<Obstacle>(new Obstacle(vss[i].template cast<T>(),(int)_obs.size())));
    else {
      *(_backup.back())=Obstacle(vss[i].template cast<T>(),(int)_obs.size());
      _obs.push_back(_backup.back());
      _backup.pop_back();
    }
  for(int i=0; i<(int)vss.size(); i++)
    _obs[offset+i]->_next=_obs[offset+(i+1)%(int)vss.size()];
}
template <typename T2>
void BoundingVolumeHierarchy::removeDuplicateVertices(std::vector<Eigen::Matrix<T2,2,1>>& vss) {
  bool more=true;
  while(more) {
    more=false;
    for(int i=0; i<(int)vss.size();) {
      Eigen::Matrix<T2,2,1> next=vss[(i+1)%(int)vss.size()]-vss[i];
      Eigen::Matrix<T2,2,1> last=vss[(i+(int)vss.size()-1)%(int)vss.size()]-vss[i];
      if(last[0]*next[1]==last[1]*next[0]) {
        vss.erase(vss.begin()+i);
        more=true;
      } else i++;
    }
  }
}
}

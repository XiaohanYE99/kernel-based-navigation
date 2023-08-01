#include "BBox.h"
#include "SourceSink.h"
#include "RVO.h"
#include "MultiRVO.h"
#include "Visibility.h"
#include "MultiVisibility.h"
#include "RVOVisualizer.h"
#include <TinyVisualizer/Drawer.h>
#include <TinyVisualizer/SceneStructure.h>

#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Python.h>

using namespace RVO;
namespace py=pybind11;
using namespace DRAWER;

PYBIND11_MODULE(pyRVO, m) {
  typedef LSCALAR T;
  DECL_MAT_VEC_MAP_TYPES_T
  //BBox
  py::class_<BBox>(m,"BBox")
  .def(py::init([]() {
    return BBox();
  }))
  .def(py::init([](const Vec2T& p) {
    return BBox(p);
  }))
  .def(py::init([](const Vec2T& minC,const Vec2T& maxC) {
    return BBox(minC,maxC);
  }))
  .def(py::init([](const BBox& other) {
    return BBox(other);
  }))
  .def_static("createMM",&BBox::createMM)
  .def_static("createME",&BBox::createME)
  .def_static("createCE",&BBox::createCE)
  .def("getIntersect",&BBox::getIntersect)
  .def("getUnion",static_cast<BBox(BBox::*)(const BBox&)const>(&BBox::getUnion))
  .def("getUnion",static_cast<BBox(BBox::*)(const Vec2T&)const>(&BBox::getUnion))
  .def("getUnion",static_cast<BBox(BBox::*)(const Vec2T&,const T&)const>(&BBox::getUnion))
  .def("setIntersect",&BBox::setIntersect)
  .def("setUnion",static_cast<void(BBox::*)(const BBox&)>(&BBox::setUnion))
  .def("setUnion",static_cast<void(BBox::*)(const Vec2T&)>(&BBox::setUnion))
  .def("setUnion",static_cast<void(BBox::*)(const Vec2T&,const T&)>(&BBox::setUnion))
  .def("setPoints",static_cast<void(BBox::*)(const Vec2T&,const Vec2T&,const Vec2T&)>(&BBox::setPoints))
  .def("setPoints",static_cast<void(BBox::*)(const Vec2T&,const Vec2T&,const Vec2T&,const Vec2T&)>(&BBox::setPoints))
  .def("minCorner",&BBox::minCorner)
  .def("maxCorner",&BBox::maxCorner)
  .def("enlargedEps",&BBox::enlargedEps)
  .def("enlargeEps",&BBox::enlargeEps)
  .def("enlarged",&BBox::enlarged)
  .def("enlarge",&BBox::enlarge)
  .def("lerp",&BBox::lerp)
  .def("empty",&BBox::empty)
  .def("contain",static_cast<bool(BBox::*)(const BBox&)const>(&BBox::contain))
  .def("contain",static_cast<bool(BBox::*)(const Vec2T&)const>(&BBox::contain))
  .def("contain",static_cast<bool(BBox::*)(const Vec2T&,const T&)const>(&BBox::contain))
  .def("reset",&BBox::reset)
  .def("getExtent",&BBox::getExtent)
  .def("distTo",static_cast<T(BBox::*)(const BBox&)const>(&BBox::distTo))
  .def("distTo",static_cast<T(BBox::*)(const Vec2T&)const>(&BBox::distTo))
  .def("distToSqr",&BBox::distToSqr)
  .def("closestTo",&BBox::closestTo)
  .def("intersect",static_cast<bool(BBox::*)(const Vec2T&,const Vec2T&)const>(&BBox::intersect))
  .def("intersect",static_cast<bool(BBox::*)(const Vec2T&,const Vec2T&,T&,T&)const>(&BBox::intersect))
  .def("intersect",static_cast<bool(BBox::*)(const BBox&)const>(&BBox::intersect))
  .def("project",&BBox::project)
  .def("copy",&BBox::copy)
  .def("perimeter",&BBox::perimeter)
  .def_readwrite("minC",&BBox::_minC)
  .def_readwrite("maxC",&BBox::_maxC);
  //Trajectory
  py::class_<Trajectory>(m,"Trajectory")
  .def(py::init([]() {
    return Trajectory();
  }))
  .def(py::init([](bool recordFull,int frameId,const Vec2T& target,T r,int id) {
    return Trajectory(recordFull,frameId,target,r,id);
  }))
  .def("startFrame",&Trajectory::startFrame)
  .def("endFrame",&Trajectory::endFrame)
  .def("isFullTrajectory",&Trajectory::isFullTrajectory)
  .def("terminated",&Trajectory::terminated)
  .def("terminate",&Trajectory::terminate)
  .def("addPos",&Trajectory::addPos)
  .def("pos",static_cast<Vec2T(Trajectory::*)(int)const>(&Trajectory::pos))
  .def("pos",static_cast<Mat2XT(Trajectory::*)()const>(&Trajectory::pos))
  .def("target",&Trajectory::target)
  .def("rad",&Trajectory::rad);
  //source sink
  py::class_<SourceSink>(m,"SourceSink")
  .def(py::init([](T maxVelocity,int maxBatch,bool recordFull) {
    return SourceSink(maxVelocity,maxBatch,recordFull);
  }))
  .def("getTrajectories",&SourceSink::getTrajectories)
  .def("addSourceSink",&SourceSink::addSourceSink)
  .def("getAgentPositions",&SourceSink::getAgentPositions)
  .def("addAgents",&SourceSink::addAgents)
  .def("recordAgents",&SourceSink::recordAgents)
  .def("removeAgents",&SourceSink::removeAgents)
  .def("reset",&SourceSink::reset);
  //RVO
  py::class_<RVOSimulator,std::shared_ptr<RVOSimulator>>(m,"RVOSimulator")
  .def(py::init([](const RVOSimulator& other) {
    return std::shared_ptr<RVOSimulator>(new RVOSimulator(other));
  }))
  .def(py::init([](T d0,T gTol,T coef,T timestep,int maxIter,bool radixSort,bool useHash,const std::string& optimizer) {
    return std::shared_ptr<RVOSimulator>(new RVOSimulator(d0,gTol,coef,timestep,maxIter,radixSort,useHash,optimizer));
  }))
  .def("getUseHash",&RVOSimulator::getUseHash)
  .def("getMaxRadius",&RVOSimulator::getMaxRadius)
  .def("getMinRadius",&RVOSimulator::getMinRadius)
  .def("clearAgent",&RVOSimulator::clearAgent)
  .def("clearObstacle",&RVOSimulator::clearObstacle)
  .def("getNrObstacle",&RVOSimulator::getNrObstacle)
  .def("getNrAgent",&RVOSimulator::getNrAgent)
  .def("getAgentTargets",&RVOSimulator::getAgentTargets)
  .def("getObstacle",&RVOSimulator::getObstacle)
  .def("getAgentPositions",static_cast<Mat2XT(RVOSimulator::*)()const>(&RVOSimulator::getAgentPositions))
  .def("getAgentVelocities",static_cast<Mat2XT(RVOSimulator::*)()const>(&RVOSimulator::getAgentVelocities))
  .def("getAgentPosition",&RVOSimulator::getAgentPosition)
  .def("getAgentVelocity",&RVOSimulator::getAgentVelocity)
  .def("getAgentDVDP",&RVOSimulator::getAgentDVDP)
  .def("getAgentRadius",static_cast<T(RVOSimulator::*)(int)const>(&RVOSimulator::getAgentRadius))
  .def("getAgentId",static_cast<int(RVOSimulator::*)(int)const>(&RVOSimulator::getAgentId))
  .def("removeAgent",&RVOSimulator::removeAgent)
  .def("addAgent",&RVOSimulator::addAgent)
  .def("setAgentPosition",&RVOSimulator::setAgentPosition)
  .def("setAgentVelocity",&RVOSimulator::setAgentVelocity)
  .def("setAgentTarget",&RVOSimulator::setAgentTarget)
  .def("addObstacle",&RVOSimulator::addObstacle)
  //.def("getVisibility",&RVOSimulator::getVisibility)
  .def("buildVisibility",static_cast<void(RVOSimulator::*)(const RVOSimulator&)>(&RVOSimulator::buildVisibility))
  .def("buildVisibility",static_cast<void(RVOSimulator::*)()>(&RVOSimulator::buildVisibility))
  .def("clearVisibility",&RVOSimulator::clearVisibility)
  .def("setNewtonParameter",&RVOSimulator::setNewtonParameter)
  .def("setLBFGSParameter",&RVOSimulator::setLBFGSParameter)
  .def("setTimestep",&RVOSimulator::setTimestep)
  .def("timestep",&RVOSimulator::timestep)
  .def("optimize",&RVOSimulator::optimize)
  .def("updateAgentTargets",&RVOSimulator::updateAgentTargets)
  .def("getDXDX",&RVOSimulator::getDXDX)
  .def("getDXDV",&RVOSimulator::getDXDV);
  //MultiRVOSimulator
  py::class_<MultiRVOSimulator,std::shared_ptr<MultiRVOSimulator>>(m,"MultiRVOSimulator")
  .def(py::init([](int batchSize,T d0,T gTol,T coef,T timestep,int maxIter,bool radixSort,bool useHash,const std::string& optimizer) {
    return std::shared_ptr<MultiRVOSimulator>(new MultiRVOSimulator(batchSize,d0,gTol,coef,timestep,maxIter,radixSort,useHash,optimizer));
  }))
  .def("clearAgent",&MultiRVOSimulator::clearAgent)
  .def("clearObstacle",&MultiRVOSimulator::clearObstacle)
  .def("clearSourceSink",&MultiRVOSimulator::clearSourceSink)
  .def("getNrObstacle",&MultiRVOSimulator::getNrObstacle)
  .def("getObstacle",&MultiRVOSimulator::getObstacle)
  .def("getNrAgent",&MultiRVOSimulator::getNrAgent)
  .def("setupSourceSink",&MultiRVOSimulator::setupSourceSink)
  .def("getTrajectories",&MultiRVOSimulator::getTrajectories)
  .def("getAllTrajectories",&MultiRVOSimulator::getAllTrajectories)
  .def("addSourceSink",&MultiRVOSimulator::addSourceSink)
  .def("setAllAgentVelocities",&MultiRVOSimulator::setAllAgentVelocities)
  .def("setAllAgentBatchVelocities",&MultiRVOSimulator::setAllAgentBatchVelocities)
  .def("getAllAgentPositions",&MultiRVOSimulator::getAllAgentPositions)
  .def("getAllAgentBatchPositions",&MultiRVOSimulator::getAllAgentBatchPositions)
  .def("getAllAgentTargets",&MultiRVOSimulator::getAllAgentTargets)
  .def("getAllAgentBatchTargets",&MultiRVOSimulator::getAllAgentBatchTargets)
  .def("getAllAgentBatchIds",&MultiRVOSimulator::getAllAgentBatchIds)
  .def("getAgentPosition",&MultiRVOSimulator::getAgentPosition)
  .def("getAgentVelocity",&MultiRVOSimulator::getAgentVelocity)
  .def("getAgentDVDP",&MultiRVOSimulator::getAgentDVDP)
  .def("getAgentRadius",&MultiRVOSimulator::getAgentRadius)
  .def("addAgent",&MultiRVOSimulator::addAgent)
  .def("setAgentPosition",&MultiRVOSimulator::setAgentPosition)
  .def("setAgentVelocity",&MultiRVOSimulator::setAgentVelocity)
  .def("setAgentTarget",&MultiRVOSimulator::setAgentTarget)
  .def("addObstacle",&MultiRVOSimulator::addObstacle)
  .def("buildVisibility",&MultiRVOSimulator::buildVisibility)
  .def("clearVisibility",&MultiRVOSimulator::clearVisibility)
  .def("setNewtonParameter",&MultiRVOSimulator::setNewtonParameter)
  .def("setLBFGSParameter",&MultiRVOSimulator::setLBFGSParameter)
  .def("setTimestep",&MultiRVOSimulator::setTimestep)
  .def("timestep",&MultiRVOSimulator::timestep)
  .def("getBatchSize",&MultiRVOSimulator::getBatchSize)
  .def("getSubSimulator",static_cast<RVOSimulator&(MultiRVOSimulator::*)(int)>(&MultiRVOSimulator::getSubSimulator))
  .def("getSubSimulator",static_cast<const RVOSimulator&(MultiRVOSimulator::*)(int)const>(&MultiRVOSimulator::getSubSimulator))
  .def("optimize",&MultiRVOSimulator::optimize)
  .def("updateAgentTargets",&MultiRVOSimulator::updateAgentTargets)
  .def("getDXDX",&MultiRVOSimulator::getDXDX)
  .def("getDXDV",&MultiRVOSimulator::getDXDV);
  //ShortestPath
  py::class_<ShortestPath>(m,"ShortestPath")
  .def_readwrite("target",&ShortestPath::_target)
  .def_readwrite("last",&ShortestPath::_last)
  .def_readwrite("distance",&ShortestPath::_distance)
  .def_readwrite("maxVelocity",&ShortestPath::_maxVelocity)
  .def_readwrite("DVDP",&ShortestPath::_DVDP);
  //VisibilityGraph
  py::class_<VisibilityGraph,std::shared_ptr<VisibilityGraph>>(m,"VisibilityGraph")
  .def(py::init([](RVOSimulator& rvo) {
    return std::shared_ptr<VisibilityGraph>(new VisibilityGraph(rvo));
  }))
  .def(py::init([](RVOSimulator& rvo,const VisibilityGraph& other) {
    return std::shared_ptr<VisibilityGraph>(new VisibilityGraph(rvo,other));
  }))
  .def("lines",static_cast<std::vector<std::pair<Vec2T,Vec2T>>(VisibilityGraph::*)(const Vec2T&)const>(&VisibilityGraph::lines))
  .def("lines",static_cast<std::vector<std::pair<Vec2T,Vec2T>>(VisibilityGraph::*)(int)const>(&VisibilityGraph::lines))
  .def("findNeighbor",&VisibilityGraph::findNeighbor)
  .def("visible",&VisibilityGraph::visible)
  .def("buildShortestPath",&VisibilityGraph::buildShortestPath)
  .def("setAgentTarget",&VisibilityGraph::setAgentTarget)
  .def("removeAgent",&VisibilityGraph::removeAgent)
  .def("getNrBoundaryPoint",&VisibilityGraph::getNrBoundaryPoint)
  .def("getAgentWayPoint",static_cast<Vec2T(VisibilityGraph::*)(const ShortestPath&,const Vec2T&,T&)const>(&VisibilityGraph::getAgentWayPoint))
  .def("getAgentWayPoint",static_cast<Vec2T(VisibilityGraph::*)(RVOSimulator&,int,T&)const>(&VisibilityGraph::getAgentWayPoint))
  .def("getAgentDVDP",&VisibilityGraph::getAgentDVDP)
  .def("updateAgentTargets",&VisibilityGraph::updateAgentTargets)
  .def("getMinDistance",&VisibilityGraph::getMinDistance);
  //MultiVisibilityGraph
  py::class_<MultiVisibilityGraph,std::shared_ptr<MultiVisibilityGraph>>(m,"MultiVisibilityGraph")
  .def(py::init([](RVOSimulator& rvo) {
    return std::shared_ptr<MultiVisibilityGraph>(new MultiVisibilityGraph(rvo));
  }))
  .def(py::init([](MultiRVOSimulator& rvo) {
    return std::shared_ptr<MultiVisibilityGraph>(new MultiVisibilityGraph(rvo));
  }))
  .def("setAgentTargets",&MultiVisibilityGraph::setAgentTargets)
  .def("setAgentPositions",&MultiVisibilityGraph::setAgentPositions)
  .def("getAgentDVDPs",&MultiVisibilityGraph::getAgentDVDPs)
  .def("getMinDistance",&MultiVisibilityGraph::getMinDistance);
  //RVOPythonCallback
  py::class_<RVOPythonCallback,std::shared_ptr<RVOPythonCallback>>(m,"RVOPythonCallback")
  .def(py::init([]() {
    return std::shared_ptr<RVOPythonCallback>(new RVOPythonCallback);
  }))
  .def_readwrite("mouse",&RVOPythonCallback::_mouse)
  .def_readwrite("wheel",&RVOPythonCallback::_wheel)
  .def_readwrite("motion",&RVOPythonCallback::_motion)
  .def_readwrite("key",&RVOPythonCallback::_key)
  .def_readwrite("frame",&RVOPythonCallback::_frame)
  .def_readwrite("draw",&RVOPythonCallback::_draw)
  .def_readwrite("setup",&RVOPythonCallback::_setup);
  //RVOVisualizer
  py::class_<RVOVisualizer>(m,"RVOVisualizer")
  .def_static("drawQuad",&RVOVisualizer::drawQuad)
  .def_static("drawLine",&RVOVisualizer::drawLine)
  .def_static("drawVisibility",static_cast<void(*)(const VisibilityGraph&,const Eigen::Matrix<LSCALAR,2,1>)>(&RVOVisualizer::drawVisibility))
  .def_static("drawVisibility",static_cast<void(*)(const VisibilityGraph& graph,int id)>(&RVOVisualizer::drawVisibility))
  .def_static("clearQuad",&RVOVisualizer::clearQuad)
  .def_static("clearLine",&RVOVisualizer::clearLine)
  .def_static("getNrQuads",&RVOVisualizer::getNrQuads)
  .def_static("setNrQuads",&RVOVisualizer::setNrQuads)
  .def_static("getNrLines",&RVOVisualizer::getNrLines)
  .def_static("setNrLines",&RVOVisualizer::setNrLines)
  .def_static("drawRVO",static_cast<void(*)(float,RVOSimulator&)>(&RVOVisualizer::drawRVO))
  .def_static("drawRVO",static_cast<void(*)(float,MultiRVOSimulator&)>(&RVOVisualizer::drawRVO))
  .def_static("drawRVO",static_cast<void(*)(float,RVOSimulator&,std::shared_ptr<RVOPythonCallback>)>(&RVOVisualizer::drawRVO))
  .def_static("drawRVO",static_cast<void(*)(float,MultiRVOSimulator&,std::shared_ptr<RVOPythonCallback>)>(&RVOVisualizer::drawRVO))
  .def_static("drawRVO",static_cast<void(*)(float,const std::vector<Trajectory>&,const RVOSimulator&)>(&RVOVisualizer::drawRVO))
  .def_static("drawRVO",static_cast<void(*)(float,const std::vector<std::vector<Trajectory>>&,const MultiRVOSimulator&)>(&RVOVisualizer::drawRVO))
  .def_static("drawRVO",static_cast<void(*)(float,const std::vector<Trajectory>&,const RVOSimulator&,std::shared_ptr<RVOPythonCallback>)>(&RVOVisualizer::drawRVO))
  .def_static("drawRVO",static_cast<void(*)(float,const std::vector<std::vector<Trajectory>>&,const MultiRVOSimulator&,std::shared_ptr<RVOPythonCallback>)>(&RVOVisualizer::drawRVO));
}

/*
 * RVOSimulator.cpp
 * RVO2 Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <https://gamma.cs.unc.edu/RVO2/>
 */

#include "RVOSimulator.h"

#include "Agent.h"
#include "KdTree.h"
#include "Obstacle.h"

#include <experimental/filesystem>
#include <iostream>
#include "time.h"
#include <fstream>
#include <string>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#define WRITE_ERROR
using namespace Eigen;

namespace RVO {
bool RVOSimulator::linesearch(const VectorXd& v,const VectorXd& x, const double Ex,
                              const VectorXd& g,const VectorXd& d,
                              double& alpha,VectorXd& xNew,
                              std::function<double(const VectorXd&)> E) {
  //we want to find: x+alpha*d
  //double alphaMin=1e-6;	//this is global
  double alphaInc=1.1;
  double alphaDec=0.6;
  double c=0.1;
  VectorXd X;

  while(alpha>alphaMin) {
    double ExNew=E(xNew+alpha*d);

    if(std::isfinite(ExNew) && ExNew<Ex+c*g.dot(d)*alpha) {
      xNew+=alpha*d;
      alpha*=alphaInc;
      break;
    } else {
      alpha*=alphaDec;
    }
  }

  return alpha>alphaMin;
}
void RVOSimulator::synchronizeAgentPositions(const VectorXd& x) {
  for(size_t i=0; i<(size_t)x.size()/2; i++) {
    agents_[i]->position_=Vector2(x[i],x[i+x.size()/2]);
  }
}
double RVOSimulator::energy(const VectorXd& v, const VectorXd& x, const VectorXd& newX,
                            Eigen::Matrix<int,4,1>& nBarrier,VectorXd* g, MatrixXd* h, bool useSpatialHash, bool useSpatialHashForObstacle) {
  nBarrier.setZero();
  double f=0.5/(timeStep_*timeStep_)*(newX-(x+v*timeStep_)).squaredNorm();
  if(g)
    *g=(newX-(x+v*timeStep_))/(timeStep_*timeStep_);
  if(h) {
    h->setIdentity(x.size(),x.size());
    (*h)/=(timeStep_*timeStep_);
  }
  //synchronize agents' positions of RVO system
  synchronizeAgentPositions(newX);
  //for spatial hash, we need to update bounding boxes
  if(useSpatialHash) {
    kdTree_->updateAgentTree();
  }
  //has a global variable SpatialHash hash;
  for(size_t i=0; i<(size_t)x.size()/2; i++) {
    if(!std::isfinite(f)) //early out
      return f;
    //compute neighborIds
    double R=agents_[i]->radius_;
    std::vector<std::pair<size_t,double>> neighborIds;
    if(useSpatialHash) {
      //use KDTree
      size_t maxNeighborsOld=agents_[i]->maxNeighbors_;
      agents_[i]->maxNeighbors_=1e5;    //set maxNeighbors to be a very large value to ensure correctness
      agents_[i]->computeNeighbors();
      agents_[i]->maxNeighbors_=maxNeighborsOld;
      for(size_t k=0; k<agents_[i]->agentNeighbors_.size(); k++) {
        size_t j=agents_[i]->agentNeighbors_[k].second->id_;
        Vector2d dist(newX[i]-newX[j],newX[i+newX.size()/2]-newX[j+newX.size()/2]);
        double margin=dist.squaredNorm()-4*R*R;
        if(i<j && margin<+d0)
          neighborIds.push_back(std::make_pair(j,margin));
      }
    } else {
      //brute force
      for(size_t j=i+1; j<(size_t)x.size()/2; ++j) {
        Vector2d dist(newX[i]-newX[j],newX[i+newX.size()/2]-newX[j+newX.size()/2]);
        double margin=dist.squaredNorm()-4*R*R;
        if(i<j && margin<+d0)
          neighborIds.push_back(std::make_pair(j,margin));
      }
    }
    //compute agent neighbor energy
    for(const std::pair<size_t,double>& neighbor:neighborIds) {
      int j=neighbor.first;
      double margin=neighbor.second;
      if(!std::isfinite(f)) //early out
        return f;
      double D=0,DD=0;
      f+=clog(margin,
              g?&D:NULL,
              h?&DD:NULL,
              d0,
              coef);	//this can be infinite or nan
      if(g) {
        (*g)[i]+=D*2*(newX[i]-newX[j]);
        (*g)[j]-=D*2*(newX[i]-newX[j]);
        (*g)[i+newX.size()/2]+=D*2*(newX[i+newX.size()/2]-newX[j+newX.size()/2]);
        (*g)[j+newX.size()/2]-=D*2*(newX[i+newX.size()/2]-newX[j+newX.size()/2]);
      }
      if(h) {
        (*h)(i,i)+=2*D+DD*4*pow(newX[i]-newX[j],2);
        (*h)(i,i+newX.size()/2)+=DD*4*(newX[i]-newX[j])*(newX[i+newX.size()/2]-newX[j+newX.size()/2]);
        (*h)(i+newX.size()/2,i)+=DD*4*(newX[i]-newX[j])*(newX[i+newX.size()/2]-newX[j+newX.size()/2]);
        (*h)(i+newX.size()/2,i+newX.size()/2)+=2*D+DD*4*pow(newX[i+newX.size()/2]-newX[j+newX.size()/2],2);

        (*h)(j,j)+=2*D+DD*4*pow(newX[i]-newX[j],2);
        (*h)(j,j+newX.size()/2)+=DD*4*(newX[i]-newX[j])*(newX[i+newX.size()/2]-newX[j+newX.size()/2]);
        (*h)(j+newX.size()/2,j)+=DD*4*(newX[i]-newX[j])*(newX[i+newX.size()/2]-newX[j+newX.size()/2]);
        (*h)(j+newX.size()/2,j+newX.size()/2)+=2*D+DD*4*pow(newX[i+newX.size()/2]-newX[j+newX.size()/2],2);

        (*h)(i,j)+=-(2*D+DD*4*pow(newX[i]-newX[j],2));
        (*h)(i,j+newX.size()/2)+=-(DD*4*(newX[i]-newX[j])*(newX[i+newX.size()/2]-newX[j+newX.size()/2]));
        (*h)(i+newX.size()/2,j)+=-(DD*4*(newX[i]-newX[j])*(newX[i+newX.size()/2]-newX[j+newX.size()/2]));
        (*h)(i+newX.size()/2,j+newX.size()/2)+=-(2*D+DD*4*pow(newX[i+newX.size()/2]-newX[j+newX.size()/2],2));

        (*h)(j,i)+=-(2*D+DD*4*pow(newX[i]-newX[j],2));
        (*h)(j,i+newX.size()/2)+=-(DD*4*(newX[i]-newX[j])*(newX[i+newX.size()/2]-newX[j+newX.size()/2]));
        (*h)(j+newX.size()/2,i)+=-(DD*4*(newX[i]-newX[j])*(newX[i+newX.size()/2]-newX[j+newX.size()/2]));
        (*h)(j+newX.size()/2,i+newX.size()/2)+=-(2*D+DD*4*pow(newX[i+newX.size()/2]-newX[j+newX.size()/2],2));
      }
      nBarrier[0]++;
    }
  }
  //for other obstacle
  for(size_t i=0; i<(size_t)x.size()/2; i++) {
    if(!std::isfinite(f)) //early out
      return f;
    double R=agents_[i]->radius_;
    std::vector<const Obstacle*> neighborIds;
    if(useSpatialHashForObstacle) {
      int maxNeighbors;
      if(!useSpatialHash) {
        //when not using KDTree, we need to turn off maxNeighbors to prevent updating agentTree
        maxNeighbors=agents_[i]->maxNeighbors_;
        agents_[i]->maxNeighbors_=0;
      }
      agents_[i]->computeNeighbors();
      if(!useSpatialHash) {
        //turn on maxNeighbors again
        agents_[i]->maxNeighbors_=maxNeighbors;
      }
      for(size_t k=0; k<agents_[i]->obstacleNeighbors_.size(); k++)
        neighborIds.push_back(agents_[i]->obstacleNeighbors_[k].second);
    } else {
      for(Obstacle* obs:obstacles_)
        neighborIds.push_back(obs);
    }
    for(const Obstacle *obstacle1:neighborIds) {
      if(!std::isfinite(f)) //early out
        return f;
      const Obstacle *obstacle2 = obstacle1->nextObstacle_;

      Vector2d pos(newX[i],newX[i+newX.size()/2]);
      const Vector2d relativePosition1(obstacle1->point_.x() - pos[0],obstacle1->point_.y()-pos[1]);
      const Vector2d relativePosition2(obstacle2->point_.x() - pos[0],obstacle2->point_.y()-pos[1]);
      const double distSq1 = relativePosition1.squaredNorm();
      const double distSq2 = relativePosition2.squaredNorm();

      const double radiusSq = 2*R*R;
      const Vector2d obstacleVector(obstacle2->point_.x() - obstacle1->point_.x(),obstacle2->point_.y() - obstacle1->point_.y());
      const double s = (-relativePosition1 .dot(obstacleVector)) / obstacleVector.squaredNorm();
      const double distSqLine = (-relativePosition1 - s * obstacleVector).squaredNorm();

      if(s<0.0 && distSq1<radiusSq+d0) {
        double D=0,DD=0;
        f+=clog(distSq1-radiusSq,
                g?&D:NULL,
                h?&DD:NULL,
                d0,
                coef);	//this can be infinite or nan
        double px=obstacle1->point_.x();
        double py=obstacle1->point_.y();
        if(g) {
          (*g)[i]+=D*2*(newX[i]-px);
          (*g)[i+newX.size()/2]+=D*2*(newX[i+newX.size()/2]-py);
        }
        if(h) {
          (*h)(i,i)+=2*D+DD*4*pow(newX[i]-obstacle1->point_.x(),2);
          (*h)(i,i+newX.size()/2)+=DD*4*(newX[i]-px)*(newX[i+newX.size()/2]-py);
          (*h)(i+newX.size()/2,i)+=DD*4*(newX[i]-px)*(newX[i+newX.size()/2]-py);
          (*h)(i+newX.size()/2,i+newX.size()/2)+=2*D+DD*4*pow(newX[i+newX.size()/2]-py,2);
        }
        nBarrier[1]++;
      } else if(s>1.0f && distSq2<radiusSq+d0) {
        double D=0,DD=0;
        f+=clog(distSq2-radiusSq,
                g?&D:NULL,
                h?&DD:NULL,
                d0,
                coef*1);	//this can be infinite or nan
        double px=obstacle2->point_.x();
        double py=obstacle2->point_.y();
        if(g) {
          (*g)[i]+=D*2*(newX[i]-px);
          (*g)[i+newX.size()/2]+=D*2*(newX[i+newX.size()/2]-py);
        }
        if(h) {
          (*h)(i,i)+=2*D+DD*4*pow(newX[i]-px,2);
          (*h)(i,i+newX.size()/2)+=DD*4*(newX[i]-px)*(newX[i+newX.size()/2]-py);
          (*h)(i+newX.size()/2,i)+=DD*4*(newX[i]-px)*(newX[i+newX.size()/2]-py);
          (*h)(i+newX.size()/2,i+newX.size()/2)+=2*D+DD*4*pow(newX[i+newX.size()/2]-py,2);
        }
        nBarrier[2]++;
      } else if(s>=0.0 && s<=1.0f && distSqLine<radiusSq+d0) {
        double D=0,DD=0;
        f+=clog(distSqLine-radiusSq,
                g?&D:NULL,
                h?&DD:NULL,
                d0,
                coef*1);	//this can be infinite or nan
        double dx1=pos[0]-obstacle1->point_.x();
        double dy1=pos[1]-obstacle1->point_.y();
        double dx=obstacleVector.x();
        double dy=obstacleVector.y();
        double cr=dx1*dy-dy1*dx;
        double len=obstacleVector.squaredNorm();
        if(g) {
          (*g)[i]+=(D*2*cr*dy)/len;
          (*g)[i+newX.size()/2]+=(-D*2*cr*dx)/len;
        }
        if(h) {
          (*h)(i,i)+=2*dy*dy*D/len+DD*pow(2*dy*cr,2)/pow(len,2);
          (*h)(i,i+newX.size()/2)+=(DD*4*dx*dy*cr*cr)/pow(len,2);
          (*h)(i+newX.size()/2,i)+=(DD*4*dx*dy*cr*cr)/pow(len,2);
          (*h)(i+newX.size()/2,i+newX.size()/2)+=2*dx*dx*D/len+DD*pow(2*dx*cr,2)/pow(len,2);
        }
        nBarrier[3]++;
      }
    }
  }
  return f;
}
bool RVOSimulator::optimize(const VectorXd& v, const VectorXd& x, VectorXd& newX, bool require_grad, bool useSpatialHash, bool output) {
  newX=x;
  VectorXd g;
  VectorXd g2;
  MatrixXd h;
  int iter;
  double alpha=1;
  Eigen::Matrix<int,4,1> nBarrier;
  double maxPerturbation=1e2;
  double minPertubation=1e-9;
  double perturbation=1e0;
  double perturbationDec=0.8;
  double perturbationInc=2.0;
  Eigen::SimplicialLDLT<SMat> sol;
  double lastAlpha;
  bool succ;
  if(useSpatialHash) {
    //build for the first time, and then only update
    synchronizeAgentPositions(x);
    kdTree_->buildAgentTree();
  }
  for(iter=0; iter<maxIter && alpha>alphaMin && perturbation<maxPerturbation; iter++) {
    //always use spatial hash to compute obstacle neighbors, but only use spatial hash to compute agent neighbors
    double E=energy(v,x,newX,nBarrier,&g,&h,useSpatialHash,true);
    if(g.cwiseAbs().maxCoeff()<tol) {
      if(output)
        std::cout<<"Exit on gNormInf<"<<tol<<std::endl;
      break;
    }
    if(iter==0) {
      maxPerturbation*=std::max(1.0,h.cwiseAbs().maxCoeff());
      minPertubation*=std::max(1.0,h.cwiseAbs().maxCoeff());
      perturbation*=std::max(1.0,h.cwiseAbs().maxCoeff());
    }
    if(output)
      std::cout << "iter=" << iter
                << " alpha=" << alpha
                << " E=" << E
                << " gNormInf=" << g.cwiseAbs().maxCoeff()
                << " perturbation=" << perturbation
                << " minPertubation=" << minPertubation <<std::endl;
    //outer-loop of line search and newton direction computation
    while(true) {
      //ensure hessian factorization is successful
      while(perturbation<maxPerturbation) {
        sol.compute((MatrixXd::Identity(x.size(), x.size())*perturbation+h).sparseView());
        if(sol.info()==Eigen::Success) {
          //perturbation=std::max(perturbation*perturbationDec,minPertubation);
          break;
        } else {
          perturbation*=perturbationInc;
        }
      }
      if(perturbation>=maxPerturbation) {
        if(output)
          std::cout<<"Exit on perturbation>=maxPerturbation"<<std::endl;
        break;
      }
      //line search
      lastAlpha=alpha;
      //VectorXd tmp=-invH.solve(g);
      succ=linesearch(v,x,E,g,-sol.solve(g),alpha,newX,[&](const VectorXd& evalPt)->double{
        return energy(v,x,evalPt,nBarrier,NULL,NULL,useSpatialHash,true);
      });
      if(succ) {
        perturbation=std::max(perturbation*perturbationDec,minPertubation);
        break;
      }
      //probably we need more perturbation to h
      perturbation*=perturbationInc;
      alpha=lastAlpha;
      if(output)
        std::cout<<"Increase perturbation to "<<perturbation<<std::endl;
    }
  }
  if(require_grad) {
    sol.compute(h.sparseView());
    partialxStar_v=sol.solve(MatrixXd::Identity(x.size(),x.size())*(1.0/timeStep_));
    partialxStar_x=sol.solve(MatrixXd::Identity(x.size(),x.size())*(1.0/(timeStep_*timeStep_)));
  }
  succ=iter<maxIter && alpha>alphaMin && perturbation<maxPerturbation;
#ifdef WRITE_ERROR
  if(!succ) {
    std::ofstream os("err.dat",std::ios::binary);
    size_t n=v.size();
    os.write((char*)&n,sizeof(size_t));
    os.write((char*)v.data(),sizeof(double)*n);
    os.write((char*)x.data(),sizeof(double)*n);
    os.write((char*)newX.data(),sizeof(double)*n);
    std::cout<<"Wrote error to err.dat"<<std::endl;
  }
#endif
  if(output)
    std::cout<<"status="<<succ<<std::endl;
  return succ;
}
void RVOSimulator::replayError() {
  if(!std::experimental::filesystem::exists("err.dat"))
    return;
  std::ifstream is("err.dat",std::ios::binary);
  size_t n;
  is.read((char*)&n,sizeof(size_t));
  VectorXd v,x,newX;
  v.resize(n);
  x.resize(n);
  newX.resize(n);
  is.read((char*)v.data(),sizeof(double)*n);
  is.read((char*)x.data(),sizeof(double)*n);
  is.read((char*)newX.data(),sizeof(double)*n);
  std::cout<<"Read from to err.dat"<<std::endl;

  Eigen::Matrix<int,4,1> nBarrier,nBarrier2;
  double delta=1e-8,f,f2;
  VectorXd g,g2,dx;
  MatrixXd h,h2;
  dx.setRandom(n);
  //test spatial hash
  f=energy(v,x,newX,nBarrier,&g,&h,false,false);
  synchronizeAgentPositions(newX);
  kdTree_->buildAgentTree();
  f2=energy(v,x,newX,nBarrier2,&g2,&h2,true,true);
  std::cout << "nBarrier=" << nBarrier.transpose() << " SpatialHash error: " << (nBarrier-nBarrier2).transpose() << std::endl;
  std::cout << "Energy  =" << f << " SpatialHash error: " << (f2-f) << std::endl;
  std::cout << "Gradient=" << g.cwiseAbs().maxCoeff() << " SpatialHash error: " << (g2-g).cwiseAbs().maxCoeff() << std::endl;
  std::cout << "Hessian =" << h.cwiseAbs().maxCoeff() << " SpatialHash error: " << (h2-h).cwiseAbs().maxCoeff() << std::endl;
  //test finite difference
  f2=energy(v,x,newX+dx*delta,nBarrier,&g2,NULL,false,false);
  std::cout << "Gradient=" << f << " FD error: " << g.dot(dx)-(f2-f)/delta << std::endl;
  std::cout << "Hessian =" << (h*dx).cwiseAbs().maxCoeff() << " FD error: " << (h*dx-(g2-g)/delta).cwiseAbs().maxCoeff() << std::endl;
  exit(EXIT_SUCCESS);
}
void RVOSimulator::checkEnergyFD(double d0Tmp_, double vScale_, double xScale_) {
  //user typically wants a larger d0 to allow more barreirs
  double d0Old=d0;
  if(d0Tmp_>0)
    d0=d0Tmp_;
  //check energy
  Eigen::Matrix<int,4,1> nBarrier,nBarrier2;
  double delta=1e-8,f,f2;
  size_t N=agents_.size();
  VectorXd v;
  VectorXd x,dx;
  VectorXd newX,newX1;
  VectorXd g,g2;
  MatrixXd h,h2;
  while(true) {
    //find valid configuration
    v.setRandom(N*2);
    x.setRandom(N*2);
    dx.setRandom(N*2);
    v*=vScale_;
    x*=xScale_;
    newX=x;
    newX1=x;
    f=energy(v,x,newX,nBarrier,NULL,NULL,false,false);
    if(!std::isfinite(f))
      continue;
    if((nBarrier.array()<=0).any())
      continue;
    //test spatial hash
    f=energy(v,x,newX,nBarrier,&g,&h,false,false);
    synchronizeAgentPositions(newX);
    kdTree_->buildAgentTree();
    f2=energy(v,x,newX,nBarrier2,&g2,&h2,true,true);
    std::cout << "nBarrier=" << nBarrier.transpose() << " SpatialHash error: " << (nBarrier-nBarrier2).transpose() << std::endl;
    std::cout << "Energy  =" << f << " SpatialHash error: " << (f2-f) << std::endl;
    std::cout << "Gradient=" << g.cwiseAbs().maxCoeff() << " SpatialHash error: " << (g2-g).cwiseAbs().maxCoeff() << std::endl;
    std::cout << "Hessian =" << h.cwiseAbs().maxCoeff() << " SpatialHash error: " << (h2-h).cwiseAbs().maxCoeff() << std::endl;
    //test finite difference
    f2=energy(v,x,newX+dx*delta,nBarrier,&g2,NULL,true,true);
    std::cout << "Gradient=" << f << " FD error: " << g.dot(dx)-(f2-f)/delta << std::endl;
    std::cout << "Hessian =" << (h*dx).cwiseAbs().maxCoeff() << " FD error: " << (h*dx-(g2-g)/delta).cwiseAbs().maxCoeff() << std::endl;
    break;
  }
  delta=1e-4;
  bool succ=optimize(v,x,newX,true,false,true);
  bool succD=optimize(v+dx*delta,x,newX1,true,false,true);
  std::cout << "Implicit-derivative=" << (partialxStar_v*dx).cwiseAbs().maxCoeff() << " Vstar error: " << ((newX1-newX)/delta-partialxStar_v*dx).squaredNorm() << std::endl;
  //recover d0
  if(d0Tmp_>0)
    d0=d0Old;
  if(!succ || !succD)
    exit(EXIT_FAILURE);
}
bool RVOSimulator::doNewtonStep(bool require_grad, bool useSpatialHash, bool output) {
  clock_t start,end;
  if(output)
    start=clock();
  size_t agent_size=static_cast<int>(agents_.size());
  VectorXd v(2*agent_size),x(2*agent_size),xNew(2*agent_size);

  for(size_t i=0; i<agent_size; i++) {
    v[i] = agents_[i]->prefVelocity_.x();
    v[i+agent_size] = agents_[i]->prefVelocity_.y();
    xNew[i] = x[i] = agents_[i]->position_.x();
    xNew[i+agent_size] = x[i+agent_size] = agents_[i]->position_.y();
  }

  bool succ=optimize(v, x, xNew, require_grad, useSpatialHash, output);
  for(size_t i=0; i<(size_t)agents_.size(); ++i) {
    agents_[i]->position_=Vector2(xNew[i],xNew[i+agent_size]);
  }
  if(output) {
    end=clock();
    std::cout << "time=" << (double)(end-start)/CLOCKS_PER_SEC << std::endl;
  }
  return succ;
}
void RVOSimulator::setNewtonParameters(size_t maxIter_, double tol_, double d0_, double coef_, double alphaMin_) {
  maxIter=maxIter_;
  tol=tol_;
  d0=d0_;
  coef=coef_;
  alphaMin=alphaMin_;
}
const Eigen::MatrixXd& RVOSimulator::getGradV() const {
  return partialxStar_v;
}
const Eigen::MatrixXd& RVOSimulator::getGradX() const {
  return partialxStar_x;
}
}

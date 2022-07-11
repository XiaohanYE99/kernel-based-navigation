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

#include <iostream>
#include "time.h"
#include <fstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>

#endif
//#define USE_SPATIAL_HASH
#define OBS
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

double RVOSimulator::energy(const VectorXd& v, const VectorXd& x, const VectorXd& newX,
                            int& nBarrier,VectorXd* g, MatrixXd* h) {

  nBarrier=0;
  double f=0.5/(timeStep_*timeStep_)*(newX-(x+v*timeStep_)).squaredNorm();
  if(g)
    *g=(newX-(x+v*timeStep_))/(timeStep_*timeStep_);
  if(h) {
    h->setIdentity(x.size(),x.size());
    (*h)/=(timeStep_*timeStep_);
  }
  //for other agent
#ifdef USE_SPATIAL_HASH
  //has a global variable SpatialHash hash;
  for(size_t i=0; i<x.size()/2; i++) {
    //agents_[i]->computeNeighbors();
    for (size_t k = 0; k < agents_[i]->agentNeighbors_.size(); k++) {
      size_t j=agents_[i]->agentNeighbors_[k].second->id_;
      if(i>=j)
        continue;
#else

  for(int i=0; i<x.size()/2; i++) {
    for (int j = i+1; j < x.size()/2; ++j) {
#endif
      double R=agents_[i]->radius_;
      Vector2d dist(newX[i]-newX[j],newX[i+newX.size()/2]-newX[j+newX.size()/2]);
      if(dist.squaredNorm()<4*R*R+d0) {
        double D,DD;
        f+=clog(dist.squaredNorm()-4*R*R,
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
        nBarrier++;
      }
    }
  }
  //for other obstacle
#ifdef OBS

  for(size_t i=0; i<x.size()/2; i++) {
    double R=agents_[i]->radius_;
    for (size_t k = 0; k < agents_[i]->obstacleNeighbors_.size(); k++) {
      const Obstacle *obstacle1 = agents_[i]->obstacleNeighbors_[k].second;
      const Obstacle *obstacle2 = obstacle1->nextObstacle_;

      Vector2d pos(newX[i],newX[i+newX.size()/2]);
      const Vector2d relativePosition1(obstacle1->point_.x() - pos[0],obstacle1->point_.y()-pos[1]);
      const Vector2d relativePosition2(obstacle2->point_.x() - pos[0],obstacle2->point_.y()-pos[1]);
      const double distSq1 = relativePosition1.squaredNorm();
      const double distSq2 = relativePosition2.squaredNorm();

      const double radiusSq = R*R;

      const Vector2d obstacleVector(obstacle2->point_.x() - obstacle1->point_.x(),obstacle2->point_.y() - obstacle1->point_.y());
      const double s = (-relativePosition1 .dot(obstacleVector)) / obstacleVector.squaredNorm();
      const double distSqLine = (-relativePosition1 - s * obstacleVector).squaredNorm();

      if (s < 0.0 && distSq1 < radiusSq+d0) {

        double D,DD;
        f+=clog(distSq1-radiusSq,
                g?&D:NULL,
                h?&DD:NULL,
                d0,
                coef*100);	//this can be infinite or nan
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
      }

      else if (s > 1.0f && distSq2 < radiusSq+d0) {

        double D,DD;
        f+=clog(distSq2-radiusSq,
                g?&D:NULL,
                h?&DD:NULL,
                d0,
                coef*100);	//this can be infinite or nan
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
      }

      else if (s >= 0.0 && s <= 1.0f && distSqLine < radiusSq+d0) {

        double D,DD;
        f+=clog(distSqLine-radiusSq,
                g?&D:NULL,
                h?&DD:NULL,
                d0,
                coef*100);	//this can be infinite or nan
        double px=obstacle1->point_.x()+s*obstacleVector.x();
        double py=obstacle1->point_.y()+s*obstacleVector.y();
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
      }
    }
  }
#endif
  return f;
}
bool RVOSimulator::optimize(const VectorXd& v, const VectorXd& x, VectorXd& newX, bool require_grad) {
  newX=x;
  VectorXd g;
  VectorXd g2;
  MatrixXd h;
  double alpha=1;
  int nBarrier,iter;
  double maxPerturbation=1e2;
  double minPertubation=1e-9;
  double perturbation=1e-8;
  double perturbationDec=0.8;
  double perturbationInc=2.0;
  //Eigen::SimplicialLDLT<Eigen::SparseMatrix<double,0,int>> invH,invB;
  //Eigen::SimplicialCholesky<SMat> sol;
  Eigen::SimplicialLDLT<SMat> sol;
  double lastAlpha;
  bool succ;
  //IF TEST
  /*VectorXd dx,dv;
  double delta=1e-8;
  dx.setRandom(x.size());
  dv.setRandom(x.size());
  double f=energy(v,x,newX,nBarrier,&g,&h);

  double f2=energy(v,x,newX+dx*delta,nBarrier,&g2,NULL);
  std::cout << "Gradient: " << g.dot(dx) <<
  " Error: " << (f2-f)/delta -g.dot(dx)<< std::endl;
  std::cout << "Hessian: " << (h*dx).norm() <<
  " Error: " << (h*dx-(g2-g)/delta).norm() << std::endl;*/
  for(iter=0; iter<maxIter && alpha>alphaMin && perturbation<maxPerturbation; iter++) {

    double E=energy(v,x,newX,nBarrier,&g,&h);

    if(g.cwiseAbs().maxCoeff()<tol) {
      //std::cout<<"Exit on gNormInf<"<<tol<<std::endl;
      break;
    }

    if(iter==0) {
      maxPerturbation*=std::max(1.0,h.cwiseAbs().maxCoeff());
      minPertubation*=std::max(1.0,h.cwiseAbs().maxCoeff());
      perturbation*=std::max(1.0,h.cwiseAbs().maxCoeff());
    }
    /*std::cout << "iter=" << iter << " alpha=" << alpha << " E=" << E << " gNormInf=" << g.cwiseAbs().maxCoeff()
    <<" perturbation=" <<perturbation<<" minPertubation=" << minPertubation <<std::endl;*/
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
        //std::cout<<"Exit on perturbation>=maxPerturbation"<<std::endl;
        break;
      }

      //line search
      lastAlpha=alpha;

      //VectorXd tmp=-invH.solve(g);

      succ=linesearch(v,x,E,g,-sol.solve(g),alpha,newX,[&](const VectorXd& evalPt)->double{
        return energy(v,x,evalPt,nBarrier,NULL,NULL);
      });

      if(succ) {
        perturbation=std::max(perturbation*perturbationDec,minPertubation);
        break;
      }

      //probably we need more perturbation to h
      perturbation*=perturbationInc;
      alpha=lastAlpha;
      //std::cout<<"Increase perturbation to "<<perturbation<<std::endl;
    }
  }
  if(require_grad) {
    sol.compute(h.sparseView());
    partialxStar_v=sol.solve(MatrixXd::Identity(x.size(), x.size())*(1.0/timeStep_));
    partialxStar_x=sol.solve(MatrixXd::Identity(x.size(), x.size())*(1.0/(timeStep_*timeStep_)));
    //if(partialxStar_x.cwiseAbs().maxCoeff()>10)
    //std::cout<<h.cwiseAbs().maxCoeff()<<"  "<<sol.solve(MatrixXd::Identity(x.size(), x.size())).cwiseAbs().maxCoeff()<<std::endl;
  }
  succ=iter<maxIter && alpha>alphaMin && perturbation<maxPerturbation;
  //std::cout<<"status="<<succ<<std::endl;
  return succ;
}

void RVOSimulator::checkEnergyFD() {
  std::ofstream fout;
  std::string filename= "/homeprintf/yxhan/yxh/kernel-based-navigation-master/hash.txt" ;
  fout.open(filename.c_str(),std::ios::out|std::ios::app);
#ifdef USE_SPATIAL_HASH
#undef USE_SPATIAL_HASH
#endif
  int N=static_cast<int>(agents_.size());
  VectorXd v;
  VectorXd x,dx;
  VectorXd newX,newX1;
  VectorXd g,g2;
  MatrixXd h;
  while(true) {

    v.setRandom(N*2);
    x.setRandom(N*2);
    dx.setRandom(N*2);

    v*=200;
    x*=100;
    dx*=1;
    newX=x;
    newX1=x;
    int nBarrier;
    double f=energy(v,x,newX,nBarrier,&g,&h);

    if(!std::isfinite(f))
      continue;
    if(nBarrier<=0)
      continue;

    break;

  }
#ifndef USE_SPATIAL_HASH
#define USE_SPATIAL_HASH
  std::vector<RVO::Vector2> goals;
  for (size_t i = 0; i < 100; ++i) {
    setAgentPosition(i,Vector2(x[i],x[i+x.size()/2]));
  }
  for (int i = 0; i < static_cast<int>(getNumAgents()); ++i) {
    setAgentPrefVelocity(i, Vector2(v[i],v[i+x.size()/2]));
  }
  kdTree_->buildAgentTree();
  for(size_t i=0; i<N; i++)
    agents_[i]->computeNeighbors();
#endif
  optimize(v, x, newX,true);
  MatrixXd q=partialxStar_v;
  double delta=1e-4;
  optimize(v+dx*delta,x,newX1,true);
  std::cout<<"Vstar error: "<<((newX1-newX)/delta-q*dx).squaredNorm()<<std::endl;
  //fout<<"newX error: "<<newX.norm()<<std::endl;

}
void RVOSimulator::doNewtonStep(bool require_grad) {
  clock_t start,end;
  start=clock();
  size_t agent_size=static_cast<int>(agents_.size());
  VectorXd v(2*agent_size),x(2*agent_size),xNew(2*agent_size);

  for(size_t i=0; i<agent_size; i++) {
    v[i]=agents_[i]->prefVelocity_.x();
    v[i+agent_size]=agents_[i]->prefVelocity_.y();
    xNew[i]=x[i]=agents_[i]->position_.x();
    xNew[i+agent_size]=x[i+agent_size]=agents_[i]->position_.y();
  }
#ifdef USE_SPATIAL_HASH
  kdTree_->buildAgentTree();

  for(size_t i=0; i<agent_size; i++)
    agents_[i]->computeNeighbors();
#endif
  optimize(v,x,xNew,require_grad);
  VectorXd dx;
  dx.setRandom(x.size());
  VectorXd xNew1=xNew;

  optimize(v, x, xNew,require_grad);
  MatrixXd q=partialxStar_x;
  double delta=1e-4;
  optimize(v,x+dx*delta,xNew1,require_grad);
  double error=((xNew1-xNew)/delta-q*dx).squaredNorm();
  if(error>1)
  std::cout<<(q*dx).squaredNorm()<<"   "<<"Vstar error: "<<error<<std::endl;

  for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
    agents_[i]->newVelocity_=Vector2((xNew[i]-x[i])/timeStep_,(xNew[i+agent_size]-x[i+agent_size])/timeStep_);
    agents_[i]->update();
  }
  end=clock();
  //printf("time=%f\n",(double)(end-start)/CLOCKS_PER_SEC);
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

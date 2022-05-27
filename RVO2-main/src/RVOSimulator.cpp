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
#define USE_SPATIAL_HASH
using namespace Eigen;

namespace RVO {
	RVOSimulator::RVOSimulator() : defaultAgent_(NULL), globalTime_(0.0f), kdTree_(NULL), timeStep_(0.0f)
	{
		kdTree_ = new KdTree(this);

	}

	RVOSimulator::RVOSimulator(float timeStep, float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed, const Vector2 &velocity) : defaultAgent_(NULL), globalTime_(0.0f), kdTree_(NULL), timeStep_(timeStep)
	{
		kdTree_ = new KdTree(this);
		defaultAgent_ = new Agent(this);

		defaultAgent_->maxNeighbors_ = maxNeighbors;
		defaultAgent_->maxSpeed_ = maxSpeed;
		defaultAgent_->neighborDist_ = neighborDist;
		defaultAgent_->radius_ = radius;
		defaultAgent_->timeHorizon_ = timeHorizon;
		defaultAgent_->timeHorizonObst_ = timeHorizonObst;
		defaultAgent_->velocity_ = velocity;
	}

	RVOSimulator::~RVOSimulator()
	{
		if (defaultAgent_ != NULL) {
			delete defaultAgent_;
		}

		for (size_t i = 0; i < agents_.size(); ++i) {
			delete agents_[i];
		}

		for (size_t i = 0; i < obstacles_.size(); ++i) {
			delete obstacles_[i];
		}

		delete kdTree_;
	}

	size_t RVOSimulator::addAgent(const Vector2 &position)
	{
		if (defaultAgent_ == NULL) {
			return RVO_ERROR;
		}

		Agent *agent = new Agent(this);

		agent->position_ = position;
		agent->maxNeighbors_ = defaultAgent_->maxNeighbors_;
		agent->maxSpeed_ = defaultAgent_->maxSpeed_;
		agent->neighborDist_ = defaultAgent_->neighborDist_;
		agent->radius_ = defaultAgent_->radius_;
		agent->timeHorizon_ = defaultAgent_->timeHorizon_;
		agent->timeHorizonObst_ = defaultAgent_->timeHorizonObst_;
		agent->velocity_ = defaultAgent_->velocity_;

		agent->id_ = agents_.size();

		agents_.push_back(agent);

		return agents_.size() - 1;
	}

	size_t RVOSimulator::addAgent(const Vector2 &position, float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed, const Vector2 &velocity)
	{
		Agent *agent = new Agent(this);

		agent->position_ = position;
		agent->maxNeighbors_ = maxNeighbors;
		agent->maxSpeed_ = maxSpeed;
		agent->neighborDist_ = neighborDist;
		agent->radius_ = radius;
		agent->timeHorizon_ = timeHorizon;
		agent->timeHorizonObst_ = timeHorizonObst;
		agent->velocity_ = velocity;

		agent->id_ = agents_.size();

		agents_.push_back(agent);

		return agents_.size() - 1;
	}

	size_t RVOSimulator::addObstacle(const std::vector<Vector2> &vertices)
	{
		if (vertices.size() < 2) {
			return RVO_ERROR;
		}

		const size_t obstacleNo = obstacles_.size();

		for (size_t i = 0; i < vertices.size(); ++i) {
			Obstacle *obstacle = new Obstacle();
			obstacle->point_ = vertices[i];

			if (i != 0) {
				obstacle->prevObstacle_ = obstacles_.back();
				obstacle->prevObstacle_->nextObstacle_ = obstacle;
			}

			if (i == vertices.size() - 1) {
				obstacle->nextObstacle_ = obstacles_[obstacleNo];
				obstacle->nextObstacle_->prevObstacle_ = obstacle;
			}

			obstacle->unitDir_ = normalize(vertices[(i == vertices.size() - 1 ? 0 : i + 1)] - vertices[i]);

			if (vertices.size() == 2) {
				obstacle->isConvex_ = true;
			}
			else {
				obstacle->isConvex_ = (leftOf(vertices[(i == 0 ? vertices.size() - 1 : i - 1)], vertices[i], vertices[(i == vertices.size() - 1 ? 0 : i + 1)]) >= 0.0f);
			}

			obstacle->id_ = obstacles_.size();

			obstacles_.push_back(obstacle);
		}

		return obstacleNo;
	}
	bool RVOSimulator::linesearch(const VectorXd& v,const VectorXd& x, const double Ex,
				const VectorXd& g,const VectorXd& d,
				double& alpha,VectorXd& xNew,
				std::function<double(const VectorXd&)> E)
    {
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
				int& nBarrier,VectorXd* g, MatrixXd* h)
    {

        nBarrier=0;
        double f=0.5/(timeStep_*timeStep_)*(newX-(x+v*timeStep_)).squaredNorm();
        if(g)
            *g=(newX-(x+v*timeStep_))/(timeStep_*timeStep_);
        if(h)
        {
            h->setIdentity(x.size(),x.size());
            (*h)/=(timeStep_*timeStep_);
        }
        //for other agent
#ifdef _OPENMP
#pragma omp parallel for
#endif
    #ifdef USE_SPATIAL_HASH
        //has a global variable SpatialHash hash;
        for(size_t i=0;i<x.size()/2;i++)
        {
			//agents_[i]->computeNeighbors();
            for (size_t k = 0; k < agents_[i]->agentNeighbors_.size(); k++)
            {
				size_t j=agents_[i]->agentNeighbors_[k].second->id_;
                if(i>=j)
                    continue;
    #else
		
        for(int i=0;i<x.size()/2;i++)
        {
            for (int j = i+1; j < x.size()/2; ++j)
            {
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
		for(size_t i=0;i<x.size()/2;i++)
        {
			double R=agents_[i]->radius_;
            for (size_t k = 0; k < agents_[i]->obstacleNeighbors_.size(); k++)
            {
				const Obstacle *obstacle1 = agents_[i]->obstacleNeighbors_[k].second;
				const Obstacle *obstacle2 = obstacle1->nextObstacle_;

				Vector2 pos=Vector2(newX[i],newX[i+newX.size()/2]);
				const Vector2 relativePosition1 = obstacle1->point_ - pos;
				const Vector2 relativePosition2 = obstacle2->point_ - pos;
				const float distSq1 = absSq(relativePosition1);
				const float distSq2 = absSq(relativePosition2);

				const float radiusSq = sqr(R);

				const Vector2 obstacleVector = obstacle2->point_ - obstacle1->point_;
				const float s = (-relativePosition1 * obstacleVector) / absSq(obstacleVector);
				const float distSqLine = absSq(-relativePosition1 - s * obstacleVector);

				if (s < 0.0f && distSq1 <= radiusSq) {
				/* Collision with left vertex. Ignore if non-convex. */
					double D,DD;
					f+=clog(distSq1-radiusSq,
                    g?&D:NULL,
                    h?&DD:NULL,
                    d0,
                    coef);	//this can be infinite or nan
					if(g) {
						(*g)[i]+=D*2*(newX[i]-obstacle1->point_.x());
						(*g)[i+newX.size()/2]+=D*2*(newX[i]-obstacle1->point_.y());
					}
				}

				else if (s > 1.0f && distSq2 <= radiusSq) {
				/* Collision with left vertex. Ignore if non-convex. */
					double D,DD;
					f+=clog(distSq2-radiusSq,
                    g?&D:NULL,
                    h?&DD:NULL,
                    d0,
                    coef);	//this can be infinite or nan
					if(g) {
						(*g)[i]+=D*2*(newX[i]-obstacle2->point_.x());
						(*g)[i+newX.size()/2]+=D*2*(newX[i]-obstacle2->point_.y());
					}
				}

				else if (s >= 0.0f && s <= 1.0f && distSqLine <= radiusSq) {
				/* Collision with left vertex. Ignore if non-convex. */
					double D,DD;
					f+=clog(distSq2-radiusSq,
                    g?&D:NULL,
                    h?&DD:NULL,
                    d0,
                    coef);	//this can be infinite or nan
					if(g) {
						(*g)[i]+=D*2*(newX[i]-obstacle2->point_.x());
						(*g)[i+newX.size()/2]+=D*2*(newX[i]-obstacle2->point_.y());
					}
				}
			}
		}
        return f;
    }
	bool RVOSimulator::optimize(const VectorXd& v, const VectorXd& x, VectorXd& newX)
    {
		clock_t start,end;
		start=clock();
        newX=x;
        VectorXd g;
        VectorXd g2;
        MatrixXd h;
        double alpha=1;
        int nBarrier,iter;
        double maxPerturbation=1e2;
        double minPertubation=1e-9;
        double perturbation=1;
        double perturbationDec=0.8;
        double perturbationInc=2.0;
        Eigen::LDLT<MatrixXd> invH;
        double lastAlpha;
        bool succ;
        for(iter=0; iter<maxIter && alpha>alphaMin && perturbation<maxPerturbation; iter++)
        {

            double E=energy(v,x,newX,nBarrier,&g,&h);

            if(g.cwiseAbs().maxCoeff()<tol)
            {
                std::cout<<"Exit on gNormInf<"<<tol<<std::endl;
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
                    invH=(MatrixXd::Identity(x.size(), x.size())*perturbation+h).ldlt();
                    if(invH.info()==Eigen::Success) {
                        //perturbation=std::max(perturbation*perturbationDec,minPertubation);
                        break;
                    } else {
                        perturbation*=perturbationInc;
                    }
                }
                if(perturbation>=maxPerturbation)
                {
                    std::cout<<"Exit on perturbation>=maxPerturbation"<<std::endl;
                    break;
                }

                //line search
                lastAlpha=alpha;
                succ=linesearch(v,x,E,g,-invH.solve(g),alpha,newX,[&](const VectorXd& evalPt)->double{
                    return energy(v,x,evalPt,nBarrier,NULL,NULL);
                });
                if(succ)
                {
                    perturbation=std::max(perturbation*perturbationDec,minPertubation);
                    break;
                }

                //probably we need more perturbation to h
                perturbation*=perturbationInc;
                alpha=lastAlpha;
                std::cout<<"Increase perturbation to "<<perturbation<<std::endl;
            }
        }
        //std::cout <<  iter <<"  "<<alpha<<" " <<perturbation << std::endl;
        succ=iter<maxIter && alpha>alphaMin && perturbation<maxPerturbation;
        //std::cout<<"status="<<succ<<std::endl;
		end=clock();
		printf("time=%f\n",(double)(end-start)/CLOCKS_PER_SEC);
        return succ;
    }
	void RVOSimulator::checkEnergyFD()
	{
		std::ofstream fout;
		std::string filename= "/home/yxhan/yxh/kernel-based-navigation-master/hash.txt" ;
		fout.open(filename.c_str(),std::ios::out|std::ios::app);
		#ifdef USE_SPATIAL_HASH
		#undef USE_SPATIAL_HASH
		#endif
		int N=static_cast<int>(agents_.size());
		VectorXd v;
		VectorXd x,dx;
		VectorXd newX;
		VectorXd g,g2;
		MatrixXd h;
		while(true) {

			v.setRandom(N*2);
			x.setRandom(N*2);
			dx.setRandom(N*2);

			v*=200;
			x*=100;
			dx*=100;
			newX=x;
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
		for(size_t i=0;i<N;i++)
			agents_[i]->computeNeighbors();
		#endif
		optimize(v, x, newX);
		fout<<"newX error: "<<newX.norm()<<std::endl;

	}
    void RVOSimulator::doStep()
    {
        size_t agent_size=static_cast<int>(agents_.size());
        VectorXd v(2*agent_size),x(2*agent_size),xNew(2*agent_size);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t i=0;i<agent_size;i++)
        {
            v[i]=agents_[i]->prefVelocity_.x();
            v[i+agent_size]=agents_[i]->prefVelocity_.y();
            xNew[i]=x[i]=agents_[i]->position_.x();
            xNew[i+agent_size]=x[i+agent_size]=agents_[i]->position_.y();
        }
		#ifdef USE_SPATIAL_HASH
		kdTree_->buildAgentTree();
		for(size_t i=0;i<agent_size;i++)
			agents_[i]->computeNeighbors();
		#endif
		
        optimize(v,x,xNew);
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
			agents_[i]->newVelocity_=Vector2((xNew[i]-x[i])/timeStep_,(xNew[i+agent_size]-x[i+agent_size])/timeStep_);
			agents_[i]->update();
		}
    }
	/*void RVOSimulator::doStep()
	{
		kdTree_->buildAgentTree();

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
			agents_[i]->computeNeighbors();
			agents_[i]->computeNewVelocity();
		}

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
			agents_[i]->update();
		}

		globalTime_ += timeStep_;
	}*/

	size_t RVOSimulator::getAgentAgentNeighbor(size_t agentNo, size_t neighborNo) const
	{
		return agents_[agentNo]->agentNeighbors_[neighborNo].second->id_;
	}

	size_t RVOSimulator::getAgentMaxNeighbors(size_t agentNo) const
	{
		return agents_[agentNo]->maxNeighbors_;
	}

	float RVOSimulator::getAgentMaxSpeed(size_t agentNo) const
	{
		return agents_[agentNo]->maxSpeed_;
	}

	float RVOSimulator::getAgentNeighborDist(size_t agentNo) const
	{
		return agents_[agentNo]->neighborDist_;
	}

	size_t RVOSimulator::getAgentNumAgentNeighbors(size_t agentNo) const
	{
		return agents_[agentNo]->agentNeighbors_.size();
	}

	size_t RVOSimulator::getAgentNumObstacleNeighbors(size_t agentNo) const
	{
		return agents_[agentNo]->obstacleNeighbors_.size();
	}

	size_t RVOSimulator::getAgentNumORCALines(size_t agentNo) const
	{
		return agents_[agentNo]->orcaLines_.size();
	}

	size_t RVOSimulator::getAgentObstacleNeighbor(size_t agentNo, size_t neighborNo) const
	{
		return agents_[agentNo]->obstacleNeighbors_[neighborNo].second->id_;
	}

	const Line &RVOSimulator::getAgentORCALine(size_t agentNo, size_t lineNo) const
	{
		return agents_[agentNo]->orcaLines_[lineNo];
	}

	const Vector2 &RVOSimulator::getAgentPosition(size_t agentNo) const
	{
		return agents_[agentNo]->position_;
	}

	const Vector2 &RVOSimulator::getAgentPrefVelocity(size_t agentNo) const
	{
		return agents_[agentNo]->prefVelocity_;
	}

	float RVOSimulator::getAgentRadius(size_t agentNo) const
	{
		return agents_[agentNo]->radius_;
	}

	float RVOSimulator::getAgentTimeHorizon(size_t agentNo) const
	{
		return agents_[agentNo]->timeHorizon_;
	}

	float RVOSimulator::getAgentTimeHorizonObst(size_t agentNo) const
	{
		return agents_[agentNo]->timeHorizonObst_;
	}

	const Vector2 &RVOSimulator::getAgentVelocity(size_t agentNo) const
	{
		return agents_[agentNo]->velocity_;
	}

	float RVOSimulator::getGlobalTime() const
	{
		return globalTime_;
	}

	size_t RVOSimulator::getNumAgents() const
	{
		return agents_.size();
	}

	size_t RVOSimulator::getNumObstacleVertices() const
	{
		return obstacles_.size();
	}

	const Vector2 &RVOSimulator::getObstacleVertex(size_t vertexNo) const
	{
		return obstacles_[vertexNo]->point_;
	}

	size_t RVOSimulator::getNextObstacleVertexNo(size_t vertexNo) const
	{
		return obstacles_[vertexNo]->nextObstacle_->id_;
	}

	size_t RVOSimulator::getPrevObstacleVertexNo(size_t vertexNo) const
	{
		return obstacles_[vertexNo]->prevObstacle_->id_;
	}

	float RVOSimulator::getTimeStep() const
	{
		return timeStep_;
	}

	void RVOSimulator::processObstacles()
	{
		kdTree_->buildObstacleTree();
	}

	bool RVOSimulator::queryVisibility(const Vector2 &point1, const Vector2 &point2, float radius) const
	{
		return kdTree_->queryVisibility(point1, point2, radius);
	}

	void RVOSimulator::setAgentDefaults(float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed, const Vector2 &velocity)
	{
		if (defaultAgent_ == NULL) {
			defaultAgent_ = new Agent(this);
		}

		defaultAgent_->maxNeighbors_ = maxNeighbors;
		defaultAgent_->maxSpeed_ = maxSpeed;
		defaultAgent_->neighborDist_ = neighborDist;
		defaultAgent_->radius_ = radius;
		defaultAgent_->timeHorizon_ = timeHorizon;
		defaultAgent_->timeHorizonObst_ = timeHorizonObst;
		defaultAgent_->velocity_ = velocity;
	}

	void RVOSimulator::setAgentMaxNeighbors(size_t agentNo, size_t maxNeighbors)
	{
		agents_[agentNo]->maxNeighbors_ = maxNeighbors;
	}

	void RVOSimulator::setAgentMaxSpeed(size_t agentNo, float maxSpeed)
	{
		agents_[agentNo]->maxSpeed_ = maxSpeed;
	}

	void RVOSimulator::setAgentNeighborDist(size_t agentNo, float neighborDist)
	{
		agents_[agentNo]->neighborDist_ = neighborDist;
	}

	void RVOSimulator::setAgentPosition(size_t agentNo, const Vector2 &position)
	{
		agents_[agentNo]->position_ = position;
	}

	void RVOSimulator::setAgentPrefVelocity(size_t agentNo, const Vector2 &prefVelocity)
	{
		agents_[agentNo]->prefVelocity_ = prefVelocity;
	}

	void RVOSimulator::setAgentRadius(size_t agentNo, float radius)
	{
		agents_[agentNo]->radius_ = radius;
	}

	void RVOSimulator::setAgentTimeHorizon(size_t agentNo, float timeHorizon)
	{
		agents_[agentNo]->timeHorizon_ = timeHorizon;
	}

	void RVOSimulator::setAgentTimeHorizonObst(size_t agentNo, float timeHorizonObst)
	{
		agents_[agentNo]->timeHorizonObst_ = timeHorizonObst;
	}

	void RVOSimulator::setAgentVelocity(size_t agentNo, const Vector2 &velocity)
	{
		agents_[agentNo]->velocity_ = velocity;
	}

	void RVOSimulator::setTimeStep(float timeStep)
	{
		timeStep_ = timeStep;
	}
	void RVOSimulator::setNewtonParameters(size_t maxIter_, double tol_, double d0_, double coef_, double alphaMin_)
	{
	    maxIter=maxIter_;
	    tol=tol_;
	    d0=d0_;
	    coef=coef_;
	    alphaMin=alphaMin_;
	}
}

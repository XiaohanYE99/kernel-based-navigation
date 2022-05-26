#include <iostream>
#include<random>
#include <Eigen/Dense>
#include<ctime>
#define INF 1e7

using namespace Eigen;

int agent_size=100;
int maxIter=10000;
double tol=1e-4;
double dt=0.25;
double R=1.0;//10;
double d0=1.5*50;
double coef=1e2;
double alphaMin=1e-6;
bool linesearch(const VectorXd& v,const VectorXd& x, const double Ex,
				const VectorXd& g,const VectorXd& d,
				double& alpha,VectorXd& xNew,
				std::function<double(const VectorXd&)> E);
double clog(double d,double* D,double* DD,double d0,double coef);
double energy(const VectorXd& v, const VectorXd& x, const VectorXd& newX,
				VectorXd* g, MatrixXd* h);
void checkEnergyFD(int N,double delta);
bool optimize(const VectorXd& v, const VectorXd& x, VectorXd& newX);
bool linesearch(const VectorXd& v,const VectorXd& x, const double Ex,
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
		//printf("#%.8f   %.8f\n",ExNew,Ex+c*g.dot(d)*alpha);
		if(std::isfinite(ExNew) && ExNew<Ex+c*g.dot(d)*alpha) {
                xNew+=alpha*d;
			alpha*=alphaInc;
			break;
		} else {
			alpha*=alphaDec;
		}
	}
	//xNew=X;
	return alpha>alphaMin;
}
double clog(double d,double* D,double* DD,double d0,double coef) {
  if(d<=0.0)
  {
      return std::numeric_limits<double>::quiet_NaN();
  }

  else if(d>d0) {
    if(D)
      *D=0;
    if(DD)
      *DD=0;
    return 0;
  }
  double valLog=log(d/d0);
  double valLogC=valLog*(d-d0);
  double relD=(d-d0)/d;
  if(D)
    *D=-(2*valLogC+(d-d0)*relD)*coef;
  if(DD)
    *DD=-(4*relD-relD*relD+2*valLog)*coef;
  return -valLogC*(d-d0)*coef;
}
double energy(const VectorXd& v, const VectorXd& x, const VectorXd& newX,
				int& nBarrier,VectorXd* g, MatrixXd* h)
{
    nBarrier=0;
	double f=0.5/(dt*dt)*(newX-(x+v*dt)).squaredNorm();
	if(g)
		*g=(newX-(x+v*dt))/(dt*dt);
	if(h)
    {
        h->setIdentity(x.size(),x.size());
		(*h)/=(dt*dt);
    }


	//for other agent
#ifdef USE_SPATIAL_HASH
	//has a global variable SpatialHash hash;
	for(int i=0;i<x.size()/2;i++)
	{
		std::vector<int> allj=
		hash.searchNeighbor(Vector2d(x[i],x[i+x.size()/2),sqrt(4*R*R+d0));
		for(int j:allj)
		{
			if(i>=j)
				continue;
#else
	for(int i=0;i<x.size()/2;i++)
	{
		for(int j=i+1;j<x.size()/2;j++)
		{
#endif
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
	return f;
}

void checkEnergyFD(int N,double delta=1e-8)
{
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
		double f2=energy(v,x,newX+dx*delta,nBarrier,&g2,NULL);
		std::cout << "Gradient: " << g.dot(dx) <<
		" Error: " << (f2-f)/delta -g.dot(dx)<< std::endl;
		std::cout << "Hessian: " << (h*dx).norm() <<
		" Error: " << (h*dx-(g2-g)/delta).norm() << std::endl;

		break;

	}
    std::cout<<optimize(v, x, newX);
}
bool optimize(const VectorXd& v, const VectorXd& x, VectorXd& newX)
{
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
		std::cout << "iter=" << iter << " alpha=" << alpha << " E=" << E << " gNormInf=" << g.cwiseAbs().maxCoeff()
		<<" perturbation=" <<perturbation<<" minPertubation=" << minPertubation <<std::endl;
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
	std::cout<<"status="<<succ<<std::endl;
	return succ;
}
int main()
{   /*
    VectorXd oldvel(2*agent_size),oldpos(2*agent_size),newpos(2*agent_size);
    for(int i=0;i<2*agent_size;i++)
    {
        oldvel[i]=8.0*(rand()%100);
        oldpos[i]=2.0*(rand()%100);
        newpos[i]=oldpos[i]+20;
    }
    optimize(oldvel,oldpos,newpos,iter);*/
    checkEnergyFD(agent_size);
    return 0;
}

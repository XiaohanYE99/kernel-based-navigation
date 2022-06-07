#include "Car.h"
#include <queue>
#include <set>
#include <fstream>
#include <iostream>

double Car::phiMax, Car::vDrivingMax, Car::steeringMax, Car::aDrivingMax, Car::length, Car::radius, Car::deltaPhi, Car::deltav1, Car::dtc, Car::errorDesired;
double Car::decelerateTime;
double Car::ka, Car::kv, Car::kp;
double Car::lengthHalf;
int Car::vNum, Car::phiNum;
std::vector<std::vector<std::vector<std::vector<double>>>> Car::errorLookupTable;

void Car::initCars(double length, double radius, double vDrivingMax, double steeringMax, double aDrivingMax, double phiMax, double dtc, double errorDesired, double ka, double kv, double kp) {
	Car::length = length;
	Car::radius = radius;

	Car::phiMax = phiMax;
	Car::vDrivingMax = vDrivingMax;
	Car::steeringMax = steeringMax;
	Car::aDrivingMax = aDrivingMax;

	Car::dtc = dtc;
	Car::errorDesired = errorDesired;

	//feedback gains solved from ��3+ka��2+kv��+kp
	Car::ka = ka;
	Car::kv = kv;
	Car::kp = kp;

	lengthHalf = length / 2;
	decelerateTime = vDrivingMax / aDrivingMax;
}

void Car::computeLookupTable(double timeConsumed, double errorClamped) {
	errorLookupTable.resize(vNum);
	for (auto& phivec : errorLookupTable)
	{
		phivec.resize(phiNum);
		for (auto& vxvec : phivec)
		{
			vxvec.resize(vNum);
			for (auto& vyvec : vxvec)
			{
				vyvec.resize(vNum);
			}
		}
	}

#pragma omp parallel for
	for (int iv0 = 0; iv0 < vNum; iv0++)
	{
		double v0 = -vDrivingMax + iv0 * deltav1;
		for (int iphi = 0; iphi < phiNum; iphi++)
		{
			double phi = -phiMax + iphi * deltaPhi;
			//for each desired velocity
			for (int ivx = 0; ivx < vNum; ivx++)
			{
				double vx = -vDrivingMax + ivx * deltav1;
				for (int ivy = 0; ivy < vNum; ivy++)
				{
					Car car(0, 0, 0, phi);
					car.v1 = v0;
					double vy = -vDrivingMax + ivy * deltav1;
					car.initTracking(RVO::Vector2(vx, vy));
					car.track();
					double t = 0;
					double maxError = 0;
					double error = 0;
					while (true)
					{
						car.update();
						t += Car::dtc;
						error = car.getError();
						if (error > maxError)
							maxError = error;
						else if (t > timeConsumed)
							break;
						if (maxError > errorClamped)
							break;
						car.track();
					}
					errorLookupTable[iv0][iphi][ivx][ivy] = maxError;
				}
			}
		}
	}
	//omp for
}

void Car::saveToFile() {
	std::ifstream infile("lookup");
	if (infile.good())
	{
		return;
	}
	std::ofstream file("lookup", std::ofstream::out);
	for (int iv = 0; iv < vNum; iv++)
	{
		for (int iphi = 0; iphi < phiNum; iphi++)
		{
			for (int ix = 0; ix < vNum; ix++)
			{
				for (int iy = 0; iy < vNum; iy++)
				{
					file << errorLookupTable[iv][iphi][ix][iy] << " ";
				}
			}
		}
	}
}

bool Car::readFromFile() {
	std::ifstream file("lookup", std::ifstream::in);
	if (!file.is_open())
	{
		return false;
	}
	errorLookupTable.resize(vNum);
	for (auto& phivec : errorLookupTable)
	{
		phivec.resize(phiNum);
		for (auto& vxvec : phivec)
		{
			vxvec.resize(vNum);
			for (auto& vyvec : vxvec)
			{
				vyvec.resize(vNum);
			}
		}
	}
	for (int iv = 0; iv < vNum; iv++)
	{
		for (int iphi = 0; iphi < phiNum; iphi++)
		{
			for (int ix = 0; ix < vNum; ix++)
			{
				for (int iy = 0; iy < vNum; iy++)
				{
					file >> errorLookupTable[iv][iphi][ix][iy];
				}
			}
		}
	}
	return true;
}

Car::Car(double centerx, double centery, double theta, double phi) :
	v1(1 / INTMAX_MAX),
	v2(0.f),
	center1(centerx), center2(centery), theta(theta), phi(phi),
	thetaDesired(0), vDesired1(0), vDesired2(0), xi1(0), xi2(0), zDesired1(0), zDesired2(0)
{
	backX = center1 - lengthHalf * cos(theta);
	backY = center2 - lengthHalf * sin(theta);
	initTracking(RVO::Vector2(0, 0));
}

void Car::control()
{
	if (decelerating)
	{
		double a = fmin(abs(v1 / dtc), aDrivingMax);
		v1 -= copysign(a, v1) * dtc;
	}
	else
	{
		track();
	}
}

void Car::track()
{
	const double cosTheta = cos(theta);
	const double sinTheta = sin(theta);
	const double tanPhi = tan(phi);
	//gai
	const double cosPhiSq = cos(pow(phi, 2));
	const double xi1Sq = pow(xi1, 2);//xi1 * xi1;
	const double xi1Cub = xi1Sq * xi1;

	const double zd1 = xi1 * cosTheta;
	const double zd2 = xi1 * sinTheta;
	const double zdd1 = -(xi1Sq)*tanPhi * sinTheta / length + xi2 * cosTheta;
	const double	zdd2 = xi1Sq * tanPhi * cosTheta / length + xi2 * sinTheta;

	//Ӧ�ʵ�����/��Сvdesired
	//cos��dcos��(t) +sin��dsin��(t)<1
	//double factor = cos(thetaDesired) * cosTheta + sin(thetaDesired) * sinTheta;
	//factor = 1 - abs(factor);

	double r1 = -ka * zdd1 + kv * (vDesired1 - zd1) + kp * (zDesired1 - center1);
	double r2 = -ka * zdd2 + kv * (vDesired2 - zd2) + kp * (zDesired2 - center2);
	//double r1 = -ka * zdd1 + kv * (vDesired1 - zd1) + kp * (zDesired1 - backX);
	//double r2 = -ka * zdd2 + kv * (vDesired2 - zd2) + kp * (zDesired2 - backY);

	//����xi
	double xi2d = xi1Cub * tan(pow(phi, 2)) / (pow(length, 2)) + r1 * cosTheta + r2 * sinTheta;
	xi2 += xi2d * dtc;
	xi2 = fmax(fmin(xi2, aDrivingMax), -aDrivingMax);
	xi1 += xi2 * dtc;
	xi1 = fmax(fmin(xi1, vDrivingMax), -vDrivingMax);

	//�����ٶ�
	v1 = xi1;
	v2 = -3 * xi2 * cosPhiSq * tanPhi / xi1 - length * r1 * cosPhiSq * sinTheta / xi1Sq + length * r2 * cosPhiSq * cosTheta / xi1Sq;
	v2 = fmax(fmin(v2, steeringMax), -steeringMax);
}

void Car::update() {
	//����λ��
	backX += v1 * dtc * cos(theta);
	backY += v1 * dtc * sin(theta);
	theta += v1 * tan(phi) * dtc / length;
	center1 = backX + lengthHalf * cos(theta);
	center2 = backY + lengthHalf * sin(theta);
	phi += v2 * dtc;
	phi = fmax(fmin(phi, phiMax), -phiMax);

	zDesired1 += vDesired1 * dtc;
	zDesired2 += vDesired2 * dtc;
}

void Car::initTracking(const RVO::Vector2& vd)
{
	vDesired1 = vd.x();
	vDesired2 = vd.y();
	thetaDesired = atan2(vDesired2, vDesired1);
	xi1 = v1;//xi1=v0=v1(0)
	xi2 = 0.f;
	//double thetaICR = copysign((PI / 2.f - abs(atan2(2.f, abs(tan(phi))))), phi) + theta;
	//double s1 = copysign(1, cos(thetaDesired) * cos(thetaICR) + sin(thetaDesired) * sin(thetaICR));
	//��������λ��
	zDesired1 = center1 /*- (s1 * lengthHalf) * cos(thetaDesired)*/;
	zDesired2 = center2 /*- (s1 * lengthHalf) * sin(thetaDesired)*/;

	decelerating = false;
}

void Car::initDecelerating() {
	vDesired1 = 0.f;
	vDesired2 = 0.f;
	thetaDesired = 0.f;
	decelerating = true;
}


#pragma once
#include "RVO.h"

class Car
{
	friend class RVOSimulator;
public:
	static double phiMax, vDrivingMax, steeringMax, aDrivingMax,
		length, radius,
		deltaPhi, deltav1, dtc, errorDesired,
		decelerateTime;
	static double lengthHalf;
	static int vNum, phiNum;
	static double ka, kv, kp;
	static std::vector<std::vector<std::vector<std::vector<double>>>> errorLookupTable;
public:
	static void initCars(double length, double radius, double  vDrivingMax, double steeringMax, double aDrivingMax, double phiMax, double dtc, double errorDesired, double ka, double kv, double kp);

	static void computeLookupTable(double timeConsumed, double errorClamped);

	static void saveToFile();

	static bool readFromFile();

	inline static void getVelocityIndex(double vx, double vy, int& ix, int& iy) {
		ix = (int)((vx + vDrivingMax) / deltav1);
		iy = (int)((vy + vDrivingMax) / deltav1);
	}

	inline static void getCarIndex(const Car& car, int& iv0, int& iphi) {
		iv0 = (int)((car.v1 + vDrivingMax) / deltav1);
		iphi = (int)((-abs(car.phi) + phiMax) / deltaPhi);
	}

	inline static RVO::Vector2 preferredVelocity(const RVO::Vector2& dir) {
		double t = RVO::abs(dir) / Car::vDrivingMax;

		double v = Car::aDrivingMax * t;/* / ( abs(v1) / aDrivingMax)*/;
		return fmin(vDrivingMax, v) * RVO::normalize(dir);
	}

	inline static bool inAgentORCA(const RVO::Vector2& vd, const std::vector<RVO::Line>& orcaLines_) {
		for (size_t i = 0; i < orcaLines_.size(); ++i) {
			if (det(orcaLines_[i].direction, orcaLines_[i].point - vd) > 0.01f) {
				return false;
			}
		}
		return true;
	}

public:
	Car(double x, double y, double theta, double phi);

	void control();

	void track();

	void update();

	void initTracking(const RVO::Vector2& vd);

	void initDecelerating();

	inline double getError() const {
		return sqrt(pow(zDesired1 - center1, 2) + pow(zDesired2 - center2, 2));
	}

	inline RVO::Vector2 getVelocity() {
		return RVO::Vector2(v1 * cos(theta), v1 * sin(theta));
	}

	inline RVO::Vector2 getCenter() const {
		return RVO::Vector2(center1, center2);
	}

public:
	double backX, backY, theta, phi;
	double center1, center2;
	double zDesired1, zDesired2;
	double vDesired1, vDesired2;
private:
	double thetaDesired;
	double v1, v2;
	double xi1;
	double xi2;
	bool decelerating;
	//double s1;
};




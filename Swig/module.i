%module pyrvo
%{
#include <RVO/Pragma.h>
#include <RVO/RVO.h>
#include <RVO/MultiRVO.h>
#include <RVO/Visibility.h>
#include <RVO/RVOVisualizer.h>
%}

%include "typemaps.i"
%include "std_vector.i"
%include "std_pair.i"
%include "eigen.i"
typedef float GLfloat;
typedef double LSCALAR;
%template(vectorPos) std::vector<Eigen::Matrix<double,2,1>>;
%template(vectorMat2XT) std::vector<Eigen::Matrix<double,2,-1>>;
%template(vectorMatT) std::vector<Eigen::Matrix<double,-1,-1>>;
%template(vectorVec) std::vector<Eigen::Matrix<double,-1,1>>;
%template(vectorT) std::vector<double>;
%template(vectorChar) std::vector<char>;

%eigen_typemaps(Eigen::Matrix<double,2,1>)
%eigen_typemaps(Eigen::Matrix<double,2,-1>)
%eigen_typemaps(Eigen::Matrix<double,3,1>)
%eigen_typemaps(Eigen::Matrix<double,3,-1>)
%eigen_typemaps(Eigen::Matrix<double,-1,-1>)
%eigen_typemaps(Eigen::Matrix<double,-1,1>)

%eigen_typemaps(Eigen::Matrix<float,2,1>)
%eigen_typemaps(Eigen::Matrix<float,2,-1>)
%eigen_typemaps(Eigen::Matrix<float,3,1>)
%eigen_typemaps(Eigen::Matrix<float,3,-1>)
%eigen_typemaps(Eigen::Matrix<float,-1,-1>)
%eigen_typemaps(Eigen::Matrix<float,-1,1>)

%include <RVO/Pragma.h>
%include <RVO/RVO.h>
%include <RVO/MultiRVO.h>
%include <RVO/Visibility.h>
%include <RVO/RVOVisualizer.h>

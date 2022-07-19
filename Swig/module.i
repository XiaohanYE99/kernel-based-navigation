%module pyrvo
%{
#include <RVO/Pragma.h>
#include <RVO/RVO.h>
#include <RVO/MultiRVO.h>
#include <RVO/Visualizer.h>
%}

%include "typemaps.i"
%include "std_vector.i"
%include "eigen.i"
typedef float GLfloat;
typedef double LSCALAR;
%template(vectorPos) std::vector<Eigen::Matrix<double,2,1>>;
%template(vectorMat2XT) std::vector<Eigen::Matrix<double,2,-1>>;
%template(vectorMatT) std::vector<Eigen::Matrix<double,-1,-1>>;
%eigen_typemaps(Eigen::Matrix<double,2,1>)
%eigen_typemaps(Eigen::Matrix<double,2,-1>)
%eigen_typemaps(Eigen::Matrix<double,-1,-1>)
%template(vectorChar) std::vector<char>;

%include <RVO/Pragma.h>
%include <RVO/RVO.h>
%include <RVO/MultiRVO.h>
%include <RVO/Visualizer.h>

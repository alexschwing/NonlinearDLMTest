#ifndef DNNHEADER
#define DNNHEADER

#include <numeric>
#include<algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

//#include <Eigen/Core>

//#define cimg_display 0
//#define cimg_use_jpeg
//#define cimg_use_png
//#define cimg_use_openmp
//#include "CImg.h"

#include "cuda_runtime.h"

#include "../libLSDN/LSDN.h"
#include "../libLSDN/Parameters.h"
#include "../libLSDN/Data.h"
#ifdef _MSC_VER
#include "../libLSDN/ComputationGraph.h"
#else
#include "../libLSDN/ComputationTree.h"
#endif
#include "../libLSDN/ParameterContainer.h"

#include "../libLSDN/Function_Affine.h"
#include "../libLSDN/Function_Relu.h"
#include "../libLSDN/Function_Dropout.h"
#include "../libLSDN/Function_Sigmoid.h"
#include "../libLSDN/Function_Softmax.h"
#include "../libLSDN/Function_Pool.h"
#include "../libLSDN/Function_Conv.h"
#include "../libLSDN/Function_ConvSub.h"
#include "../libLSDN/Function_Lrn.h"
#include "../libLSDN/Function_Interpolate.h"
#include "../libLSDN/Function_InterpolateToSize.h"

#include "../libLSDN/LSDN_mathfunctions.h"

//#include "../ReadDirectory.h"
#include "../CPrecisionTimer.h"
//typedef Matrix<short, Dynamic, 1> VectorXs;
using std::vector;
using std::string;
#ifdef _MSC_VER
#define WHICH_DATA_TO_USE "..\\Data\\VOC2012Demo\\"
#else
//#define WHICH_DATA_TO_USE "Data/VOC2012Demo/"
#define WHICH_DATA_TO_USE "/ais/gobi3/u/aschwing/VOC2012/VOC2012/"
#endif

#ifdef _MSC_VER
#define GPUTYPE false
#else
#define GPUTYPE false
#endif
//typedef float ValueType;
typedef double ValueType;
typedef int SizeType;
typedef Node<ValueType, SizeType, GPUTYPE> NodeType;
typedef Parameters<NodeType> ParaType;
typedef Data<NodeType> DataType;
typedef ComputeFunction<NodeType> FuncType;
typedef ComputationTree<NodeType> CompTree;

typedef vector<vector<vector<ValueType>>> vector3d;
typedef vector<vector<ValueType>> vector2d;

void initCNN(double alpha, double beta, CompTree *DeepNet16, ParameterContainer<ValueType, SizeType, GPUTYPE, false> &DeepNet16Params, const vector3d & pObjOri, const vector3d & nObjOri,
            const int D, const int N, const int GPUid = 0);

template<bool positive=false>
double performTrain(double &norm, CompTree *DeepNet16, ParameterContainer<ValueType, SizeType, GPUTYPE, false> &DeepNet16Params,
                  const int pSize, const int nSize, const double epsilon);

double perceptronTrain(double &norm, CompTree *DeepNet16, ParameterContainer<ValueType, SizeType, GPUTYPE, false> &DeepNet16Params,
                  const int pSize, const int nSize, const double epsilon);
double APSVMTrain(double &norm, CompTree *DeepNet16, ParameterContainer<ValueType, SizeType, GPUTYPE, false> &DeepNet16Params,
                  const int pSize, const int nSize);
void genData(string name, int D = 10, int N = 1000, bool picture = false, const int GPUid = 3);
void createTestNet(double alpha, double beta, CompTree* CNN, ParameterContainer<ValueType, SizeType, GPUTYPE, false>& paramContainer, DataType* dataC);
#endif // DNN


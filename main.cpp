#include <cmath>
#include <functional>
#include <iterator>
#include <iostream>
#include <numeric>
#include <fstream>
#include <random>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <cassert>
#include "dp.h"
#include "dnn.h"
#include <cmath>

using std::vector;
using std::string;
using std::cout;
using std::endl;
typedef vector<vector<vector<double>>> vector3d;
typedef vector<vector<double>> vector2d;
const int D = 10; // Dimensions of data;

void readin(string name, vector3d &pObj, vector3d &nObj){
    std::ifstream fin(name);
    assert(fin);
    int N;
    fin >> N;
    pObj.resize(N);
    nObj.resize(N);
    for(int i = 0; i < N;i++){
        int x,y;
        fin>>x>>y;
        pObj[i].resize(x);
        for(int j = 0;j < x;j++){
            pObj[i][j].resize(y);
            for(int k = 0;k < y; k++)
                fin >> pObj[i][j][k];
        }

        fin>>x>>y;
        nObj[i].resize(x);
        for(int j = 0;j < x;j++){
            nObj[i][j].resize(y);
            for(int k = 0;k < y;k++)
                fin >> nObj[i][j][k];
        }
    }
}

double test(CompTree *dnn, int pSize, int nSize){
    dnn->ForwardPass(TRAIN);

    ValueType *output = dnn->GetRoot()->GetValuePtr();
    vector<ValueType> scores(pSize+nSize,0);
    for(size_t i = 0, e = scores.size(); i != e; i++)
        scores[i] = output[i];

    vector<int> iota(pSize+nSize,0);
    std::iota(iota.begin(),iota.end(),0);
    std::sort(iota.begin(),iota.end(),[&](int a,int b){
        return (scores[a] > scores[b]) ? true:false;
    });

    int count = 0;
    double ap = 0;
    for(int i = 0; i <pSize+nSize;i++)
        if(iota[i] < pSize){
            count++;
            ap += count / (i + 1.0);
        }
    return ap/count;
}
void play(CompTree *dnn, ParameterContainer<ValueType,SizeType,GPUTYPE,false> &paramContainer){
    dnn = new CompTree;
    DataType *dnnData = new DataType(DataType::NodeParameters());
    //ValueType * testdata = new ValueType[2 * 3]{1,-1,-1,2,-1,2};
    ValueType* testdata = new ValueType[3 * 2]{1,2,-1,-1,-1,2};
    dnnData->SetValueSize(testdata, new SizeType[2]{2,3},2);
   //createTestNet(dnn, paramContainer, dnnData);

    vector<ValueType> initWeights(paramContainer.GetWeightDimension(), 1.0);

    paramContainer.SetWeights(i2t<GPUTYPE>(), &initWeights);

    dnn->ForwardPass(TRAIN);
    ValueType *output = dnn->GetRoot()->GetValuePtr();
    SizeType* outputSize = dnn->GetRoot()->GetSizePtr();
    cout << outputSize[0] << ' ' << outputSize[1] << endl;
    assert(outputSize[0] == 1);
    assert(outputSize[1] == 3);

    for(int i = 0; i != outputSize[1]; i++)
        cout << output[i] <<endl;

    ValueType* testdata2 = new ValueType[2*2]{-1,-1,-1,2};
    dnnData->SetValueSize(testdata2, new SizeType[2]{2,2},2);

    dnn->ForwardPass(TRAIN);
    cout<<"------Another data-------"<<endl;
    outputSize = dnn->GetRoot()->GetSizePtr();
    cout<<outputSize[1]<<endl;
    for(int i = 0; i != outputSize[1]; i++)
        cout << output[i] << endl;

}

int main(int argc, char *argv[])
{
    vector3d trainP, trainN, testP, testN, validP, validN;
   // readin("train100.txt",trainP,trainN);
    //readin("test100.txt",testP,testN);
    //readin("valid100.txt",validP,validN);

      //genData("10000train.txt",10,10000,false,0);
    double alpha,beta,epsilon;
    int positive;
    if(argc != 5){
        std::cerr << "Number of arguments do not match" << std::endl;
        return -1;
    }
    alpha = std::stod(argv[1]);
    beta = std::stod(argv[2]);
    epsilon = std::stod(argv[3]);
    positive = std::atoi(argv[4]);
    readin("10000train.txt",trainP,trainN);
    CompTree *dnn = new CompTree;
    ParameterContainer<ValueType,SizeType,GPUTYPE,false> dnnParams;
    //play(dnn, dnnParams);
    initCNN(alpha,beta,dnn,dnnParams,trainP,trainN,10,10000);
    //test(dnn, trainP[0].size(), trainN[0].size());
    //cout<<"Testing AP: "<<test(dnn,trainP[0].size(),trainN[0].size())<<endl;
    for(int iter = 0; iter < 500; iter ++){
        double ap = 0;
        double norm = -1;
        if(positive == 1)
            ap = performTrain<true>(norm,dnn,dnnParams,trainP[0].size(),trainN[0].size(),epsilon);
        else if(positive == -1){
            std::cout<<"In negative case"<<std::endl;
            ap = performTrain<false>(norm,dnn,dnnParams,trainP[0].size(),trainN[0].size(),epsilon);
        }
        else if(positive == 0)// no relationship to epsilon
            ap = perceptronTrain(norm,dnn,dnnParams,trainP[0].size(),trainN[0].size(),1);
        else if(positive == 2)// no relationship to epsilon
            ap = APSVMTrain(norm,dnn,dnnParams,trainP[0].size(),trainN[0].size());
//        if(iter == 500) {
//           cout << "Reduce step size here:" << endl;
//           dnnParams.ReduceStepSize();
//        }

        std::cout << "Average Precision: "<<ap<<" Norm of derivs: "<<norm<<std::endl;
    }
    return 0;
}

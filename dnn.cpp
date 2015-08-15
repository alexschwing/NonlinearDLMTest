#include "dnn.h"
#include "dp.h"
//using namespace Eigen;
#include <random>
using std::cout;
using std::endl;
CompTree::TreeSiblIter AppendAffineFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, bool bias, bool relu) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    AffineFunction<NodeType>* AffFunc = new AffineFunction<NodeType>(AffineFunction<NodeType>::NodeParameters(bias, relu));
    AffFunc->SetValueSize(NULL, NULL, 2);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(AffFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}

CompTree::TreeSiblIter AppendConvFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, bool bias, bool relu, SizeType padding) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    ConvFunction<NodeType>* ConvFunc = new ConvFunction<NodeType>(ConvFunction<NodeType>::NodeParameters(padding, 1, 1, bias, relu));
    ConvFunc->SetValueSize(NULL, NULL, 4);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(ConvFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}
CompTree::TreeSiblIter AppendConvSubFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, bool bias, bool relu, SizeType padding, SizeType SubsampleH, SizeType SubsampleW) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    ConvSubFunction<NodeType>* ConvSubFunc = new ConvSubFunction<NodeType>(ConvSubFunction<NodeType>::NodeParameters(padding, 1, 1, SubsampleH, SubsampleW, bias, relu));
    ConvSubFunc->SetValueSize(NULL, NULL, 4);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(ConvSubFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}
CompTree::TreeSiblIter AppendDropoutFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, ValueType rate, NodeType* param1, NodeType* param2) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    DropoutFunction<NodeType>* DropFunc = new DropoutFunction<NodeType>(DropoutFunction<NodeType>::NodeParameters(rate));
    DropFunc->SetValueSize(NULL, NULL, 2);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(DropFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}
CompTree::TreeSiblIter AppendPoolingFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, SizeType stride, SizeType padding, SizeType SubsampleH, SizeType SubsampleW) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    PoolFunction<NodeType>* PoolFunc = new PoolFunction<NodeType>(PoolFunction<NodeType>::NodeParameters(2, 2, padding, stride, SubsampleH, SubsampleW, PoolFunction<NodeType>::MAX_POOLING));
    PoolFunc->SetValueSize(NULL, NULL, 2);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(PoolFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}
// Another version
CompTree::TreeSiblIter AppendPoolingFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, SizeType kernelheight, SizeType kernelwidth, SizeType stride, SizeType padding, SizeType SubsampleH, SizeType SubsampleW) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    PoolFunction<NodeType>* PoolFunc = new PoolFunction<NodeType>(PoolFunction<NodeType>::NodeParameters(kernelheight, kernelwidth, padding, stride, SubsampleH, SubsampleW, PoolFunction<NodeType>::MAX_POOLING));
    PoolFunc->SetValueSize(NULL, NULL, 2);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(PoolFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}
CompTree::TreeSiblIter AppendLRNFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, SizeType length, ValueType alpha, ValueType beta) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    LrnFunction<NodeType>* LRNFunc = new LrnFunction<NodeType>(LrnFunction<NodeType>::NodeParameters(length, alpha, beta));
    LRNFunc->SetValueSize(NULL, NULL, 4);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(LRNFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}
CompTree::TreeSiblIter AppendConvFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, bool bias, bool relu, SizeType padding, SizeType stride, SizeType groups) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    ConvFunction<NodeType>* ConvFunc = new ConvFunction<NodeType>(ConvFunction<NodeType>::NodeParameters(padding, stride, groups, bias, relu));
    ConvFunc->SetValueSize(NULL, NULL, 4);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(ConvFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}

void createTestNet(double alpha, double beta, CompTree* CNN, ParameterContainer<ValueType, SizeType, GPUTYPE, false>& paramContainer, DataType* dataC){
    ValueType stepSize2 = ValueType(-alpha); //P5: ValueType(-5e-8); //P4: ValueType(-1e-6); //P3: ValueType(-1e-7); //P2: ValueType(-5e-6); //P1: ValueType(-5e-7);
    ValueType moment = ValueType(0.0);// ValueType(0.9);
    ValueType l2_reg = ValueType(beta);// ValueType(0.0005);// ValueType(1); // ValueType(0.0005);
    ValueType red = ValueType(2);

#ifdef _MSC_VER
    paramContainer.AddParameter(new SizeType[2]{10,5}, 2, ParaType::NodeParameters(stepSize2,moment,red,l2_reg,0,NULL,false),false,0);
	paramContainer.AddParameter(new SizeType[1]{5}, 1, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL, false), false, 1);
	paramContainer.AddParameter(new SizeType[2]{5, 3}, 2, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL, false), false, 2);
	paramContainer.AddParameter(new SizeType[1]{3}, 1, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL, false), false, 3);
	paramContainer.AddParameter(new SizeType[2]{3, 1}, 2, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL, false), false, 4);
#else
	paramContainer.AddParameter(new SizeType[2]{10,5}, 2, ParaType::NodeParameters(stepSize2,moment,red,l2_reg,0,NULL),false,0);
	paramContainer.AddParameter(new SizeType[1]{5}, 1, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL), false, 1);
	paramContainer.AddParameter(new SizeType[2]{5, 3}, 2, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL), false, 2);
	paramContainer.AddParameter(new SizeType[1]{3}, 1, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL), false, 3);
	paramContainer.AddParameter(new SizeType[2]{3, 1}, 2, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL), false, 4);
#endif

    paramContainer.CreateCPUMemoryForParameters();
    paramContainer.PrepareComputation(TRAIN);

    AffineFunction<NodeType>* AffFunc = new AffineFunction<NodeType>(AffineFunction<NodeType>::NodeParameters(false, false));
    AffFunc->SetValueSize(NULL, NULL, 2);

    CompTree::TreeSiblIter nodeIter = CNN->insert(AffFunc);
    nodeIter = AppendAffineFunction(CNN, nodeIter, paramContainer.GetPtrFromID(4), NULL ,true,true);
    nodeIter = AppendAffineFunction(CNN, nodeIter, paramContainer.GetPtrFromID(2),paramContainer.GetPtrFromID(3), true, true);
    CNN->append_child(paramContainer.GetPtrFromID(0), nodeIter);
    CNN->append_child(dataC, nodeIter);
    CNN->append_child(paramContainer.GetPtrFromID(1), nodeIter);
}

void SetWeights(const std::string& weightFile, ParameterContainer<ValueType, SizeType, GPUTYPE, false>& paramContainer) {
    std::vector<ValueType> initWeights(paramContainer.GetWeightDimension(), ValueType(0.0));
    std::ifstream ifs(weightFile.c_str(), std::ios_base::binary | std::ios_base::in | std::ios_base::ate);
    if (ifs.is_open()) {
        std::ifstream::pos_type FileSize = ifs.tellg();
        ifs.seekg(0, std::ios_base::beg);
        if (size_t(FileSize) == initWeights.size()*sizeof(ValueType)) {
            ifs.read((char*)&initWeights[0], initWeights.size()*sizeof(ValueType));
        } else {
            std::cout << "Dimensions of initial weight file (" << FileSize / sizeof(ValueType) << ") don't match parameter dimension (" << initWeights.size() << "). Using default." << std::endl;
        }
        ifs.close();
    } else {
        std::cout << "Could not open weight file '" << weightFile << "'. Using default." << std::endl;
    }
    if (initWeights.size() > 0) {
        paramContainer.SetWeights(i2t<GPUTYPE>(), &initWeights);
    }
}
void DerivativeTest() {
    ParameterContainer<ValueType, SizeType, GPUTYPE, false> DiffTestParams;
#ifdef _MSC_VER
    //DiffTestParams.AddParameter(new SizeType[4]{7, 7, 5, 6}, 4, ParaType::NodeParameters(-1.0, 0.0, 2, 0, 10, NULL), false, 26);
    //DiffTestParams.AddParameter(new SizeType[1]{6}, 1, ParaType::NodeParameters(-1.0, 0.0, 2, 0, 10, NULL), false, 27);
    //DiffTestParams.AddParameter(new SizeType[4]{10, 10, 5, 1}, 4, ParaType::NodeParameters(-1.0, 0.0, 2, 0, 10, NULL), false, 0);
    //DiffTestParams.AddParameter(new SizeType[2]{20, 1}, 2, ParaType::NodeParameters(-1.0, 0.0, 2, 0, 10, NULL), false, 2);
	DiffTestParams.AddParameter(new SizeType[2]{20, 20}, 2, ParaType::NodeParameters(-1.0, 0.0, 2, 0, 10, NULL, false), false, 2);
#else
	DiffTestParams.AddParameter(new SizeType[2]{20, 20}, 2, ParaType::NodeParameters(-1.0, 0.0, 2, 0, 10, NULL), false, 2);
#endif
    DiffTestParams.CreateCPUMemoryForParameters();
    DiffTestParams.PrepareComputation(TRAIN);

    std::vector<ValueType> DiffTestWeights(DiffTestParams.GetWeightDimension(), ValueType(1.0));
    srand(1);
    for (std::vector<ValueType>::iterator iter = DiffTestWeights.begin(), iter_e = DiffTestWeights.end(); iter != iter_e; ++iter) {
        *iter = ValueType(rand()) / ValueType(RAND_MAX) - ValueType(0.5);
        //*iter = std::max((iter - DiffTestWeights.begin()) % 10, ((iter - DiffTestWeights.begin()) / 10) % 10);
    }
    //DiffTestWeights[0] = ValueType(200);https://github.com/hjss06/NonlinearDLMTest.git
    DiffTestParams.SetWeights(i2t<GPUTYPE>(), &DiffTestWeights);

    ConvFunction<NodeType>* ConvTestFunc = new ConvFunction<NodeType>(ConvFunction<NodeType>::NodeParameters(1, 1, 1, true, false));
    ConvTestFunc->SetValueSize(NULL, NULL, 0);
    ConvSubFunction<NodeType>* ConvSubTestFunc = new ConvSubFunction<NodeType>(ConvSubFunction<NodeType>::NodeParameters(1, 1, 1, 4, 4, true, false));
    ConvSubTestFunc->SetValueSize(NULL, NULL, 0);
    AffineFunction<NodeType>* AffFunc = new AffineFunction<NodeType>(AffineFunction<NodeType>::NodeParameters(false, false));
    AffFunc->SetValueSize(NULL, NULL, 0);
    PoolFunction<NodeType>* PoolFunc = new PoolFunction<NodeType>(PoolFunction<NodeType>::NodeParameters(2, 2, 1, 1, 1, 1, PoolFunction<NodeType>::MAX_POOLING));
    PoolFunc->SetValueSize(NULL, NULL, 0);
    InterpolateFunction<NodeType>* IntPolFunc = new InterpolateFunction<NodeType>(InterpolateFunction<NodeType>::NodeParameters(3, 3));
    IntPolFunc->SetValueSize(NULL, NULL, 0);
    InterpolateToSizeFunction<NodeType>* IntPolSizeFunc = new InterpolateToSizeFunction<NodeType>(InterpolateToSizeFunction<NodeType>::NodeParameters(21, 21));
    IntPolSizeFunc->SetValueSize(NULL, NULL, 0);

    CompTree* DiffTest = new CompTree;
    CompTree::TreeSiblIter nodeIterTest = DiffTest->insert(AffFunc);
    DiffTest->append_child(DiffTestParams.GetPtrFromID(2), nodeIterTest);
    DiffTest->append_child(DiffTestParams.GetPtrFromID(2), nodeIterTest);
    DiffTest->append_child(DiffTestParams.GetPtrFromID(2), nodeIterTest);
    //DiffTest->append_child(DiffTestParams.GetPtrFromID(0), nodeIterTest);
    //DiffTest->append_child(DiffTestParams.GetPtrFromID(27), nodeIterTest);

    DiffTest->ForwardPass(TEST);
    SizeType numOutputEl = DiffTest->GetRoot()->GetNumEl();

    ValueType* DiffTestOutput;
    if (GPUTYPE) {
        DiffTestOutput = new ValueType[numOutputEl];
        cudaMemcpy((char*)DiffTestOutput, DiffTest->GetRoot()->GetValuePtr(), numOutputEl*sizeof(ValueType), cudaMemcpyDeviceToHost);
    } else {
        DiffTestOutput = DiffTest->GetRoot()->GetValuePtr();
    }

    ValueType SumRes = 0;
    for (SizeType k = 0; k < numOutputEl; ++k) {
        SumRes += DiffTestOutput[k];
    }

    ValueType* deriv = new ValueType[numOutputEl];
    std::fill(deriv, deriv + numOutputEl, ValueType(1.0));
    ValueType** diff = DiffTest->GetRoot()->GetDiffGradientAndEmpMean();
    ValueType* diffGPU = NULL;
    if (GPUTYPE) {
        cudaMalloc((void**)&diffGPU, sizeof(ValueType)*numOutputEl);
        cudaMemcpy(diffGPU, deriv, numOutputEl*sizeof(ValueType), cudaMemcpyHostToDevice);
        *diff = diffGPU;
    } else {
        *diff = deriv;
    }

    DiffTest->BackwardPass();

    std::vector<ValueType> ComputedDerivative;
    DiffTestParams.GetDerivative(i2t<GPUTYPE>(), ComputedDerivative);

    std::vector<ValueType> ApproximatedDerivative(ComputedDerivative.size(), ValueType(0.0));
    std::vector<ValueType> ApproximatedDerivativeOnesided(ComputedDerivative.size(), ValueType(0.0));
    ValueType norm = ValueType(0.0);
    ValueType maxAbsDiff = ValueType(0.0);
    ValueType maxAbsDiffOS = ValueType(0.0);
    size_t maxDiffDim = 0;
    ValueType h = ValueType(1e-5);
    assert(ApproximatedDerivative.size() == DiffTestWeights.size());
    for (size_t k = 0; k < ApproximatedDerivative.size(); ++k) {
        ValueType origWeight = DiffTestWeights[k];
        DiffTestWeights[k] += h;

        DiffTestParams.SetWeights(i2t<GPUTYPE>(), &DiffTestWeights);
        DiffTest->ForwardPass(TEST);
        if (GPUTYPE) {
            cudaMemcpy((char*)DiffTestOutput, DiffTest->GetRoot()->GetValuePtr(), numOutputEl*sizeof(ValueType), cudaMemcpyDeviceToHost);
        }
        ValueType f1 = 0;
        for (SizeType m = 0; m < numOutputEl; ++m) {
            f1 += DiffTestOutput[m];
        }

        DiffTestWeights[k] = origWeight - h;

        DiffTestParams.SetWeights(i2t<GPUTYPE>(), &DiffTestWeights);
        DiffTest->ForwardPass(TEST);
        if (GPUTYPE) {
            cudaMemcpy((char*)DiffTestOutput, DiffTest->GetRoot()->GetValuePtr(), numOutputEl*sizeof(ValueType), cudaMemcpyDeviceToHost);
        }
        ValueType f2 = 0;
        for (SizeType m = 0; m < numOutputEl; ++m) {
            f2 += DiffTestOutput[m];
        }

        DiffTestWeights[k] = origWeight;

        ApproximatedDerivative[k] = (f1 - f2) / (2 * h);
        ApproximatedDerivativeOnesided[k] = (f1 - SumRes) / h;

        ValueType diff = ComputedDerivative[k] - ApproximatedDerivative[k];
        norm += (diff)*(diff);
        if (fabs(diff) > maxAbsDiff) {
            maxAbsDiff = fabs(diff);
            maxDiffDim = k;
        }

        ValueType diffOS = ComputedDerivative[k] - ApproximatedDerivativeOnesided[k];
        if (fabs(diffOS) > maxAbsDiffOS) {
            maxAbsDiffOS = fabs(diffOS);
        }
    }

    std::cout << "Norm of deriv diff:           " << std::sqrt(norm) << std::endl;
    std::cout << "Max abs value deriv diff:     " << maxAbsDiff << std::endl;
    std::cout << "Max abs value deriv diff dim: " << maxDiffDim << std::endl;
}
void deleteCNN(CompTree *DeepNet16, ParameterContainer<ValueType, SizeType, GPUTYPE, false> &DeepNet16Params){
    std::set<NodeType*> nothing;
    DeepNet16->Clear(&nothing, true);
    for (std::set<NodeType*>::iterator it = nothing.begin(), it_e = nothing.end(); it != it_e; ++it) {
        delete *it;
    }
    DeepNet16Params.Clear();
    delete DeepNet16;
}


void initCNN(double alpha, double beta, CompTree *DeepNet16, ParameterContainer<ValueType, SizeType, GPUTYPE, false> &DeepNet16Params,
            const vector3d & pObjOri, const vector3d & nObjOri,
            const int D, const int N, const int GPUid ) {

    if (GPUTYPE) {
        int GPUboard = GPUid;
        if (cudaSetDevice(GPUboard) != cudaSuccess) {
            std::cout << "Cannot set GPU device " << GPUboard << std::endl;
            return;
        } else {
            std::cout << "Using GPU " << GPUboard << std::endl;
        }
    } else {
        std::cout << "NOT using GPU. Are you sure?" << std::endl;
    }

    //LSDN::Instance().SetSeed(1);

    DataType * DeepNet16Data = new DataType(DataType::NodeParameters());

    ValueType* dataPtr = new ValueType[D*N]();
    int pSize = pObjOri[0].size(), nSize = nObjOri[0].size();
    for(int i = 0; i != pSize; i++)
        for(int j = 0; j != D; j++)
            dataPtr[i * D + j] = pObjOri[0][i][j];
    for(int i = 0; i != nSize; i++)
        for(int j = 0; j != D; j++)
            dataPtr[(i + pSize) * D + j] = nObjOri[0][i][j];

    ValueType* dataPtrGPU = NULL;

    if (GPUTYPE) {
        cudaMalloc((void**)&dataPtrGPU, sizeof(ValueType)*D*N);
        cudaMemcpy((char*)dataPtrGPU, (char*)dataPtr, sizeof(ValueType)*D*N, cudaMemcpyHostToDevice);
        DeepNet16Data->SetValueSize(dataPtrGPU, new SizeType[2]{D,N}, 2);
    } else {
        DeepNet16Data->SetValueSize(dataPtr, new SizeType[2]{D,N}, 2);
    }

    createTestNet(alpha, beta, DeepNet16, DeepNet16Params, DeepNet16Data);

    vector<ValueType> initWeights(DeepNet16Params.GetWeightDimension(), ValueType(0.0));
    std::random_device rd;
    std::mt19937 eng(rd());
    std::normal_distribution<double> dist(0,1);
    for(auto &v: initWeights) v = dist(eng);

    DeepNet16Params.SetWeights(i2t<GPUTYPE>(),&initWeights);

    //DeepNet16->ForwardPass(TRAIN);
//    ValueType *output = DeepNet16->GetRoot()->GetValuePtr();
//    cout<<"-----scores in init------"<<endl;
//    for(int i = 0;i < N; i++)
//        cout<<output[i]<<endl;
//#ifdef _MSC_VER
//    std::string InitWeights("..\\InitialWeights.dat");
//#else
//    std::string InitWeights("InitialWeights.dat");
//#endif
//    SetWeights(InitWeights, DeepNet16Params);

}

template<bool positive=false>
double performTrain(double &norm, CompTree *DeepNet16, ParameterContainer<ValueType, SizeType, GPUTYPE, false> &DeepNet16Params,
                  const int pSize, const int nSize, const double epsilon){

    DeepNet16->ForwardPass(TRAIN);
    ValueType* DeepNet16output;
    NodeType* rootNodeDeepNet = DeepNet16->GetRoot();
    if (GPUTYPE) {
        DeepNet16output = new ValueType[rootNodeDeepNet->GetNumEl()];
        cudaMemcpy((char*)DeepNet16output, rootNodeDeepNet->GetValuePtr(), rootNodeDeepNet->GetNumEl()*sizeof(ValueType), cudaMemcpyDeviceToHost);
    } else {
        DeepNet16output = DeepNet16->GetRoot()->GetValuePtr();
    }

    SizeType* outputDimensions = rootNodeDeepNet->GetSizePtr();
//    assert(outputDimensions[1] == 2);
    assert(outputDimensions[0] == 1);

    vector<double> scores(pSize+nSize,0);
    for(size_t i = 0; i != pSize; i++)  scores[i] = DeepNet16output[i];
    for(size_t i = 0; i != nSize; i++)  scores[pSize + i] = DeepNet16output[pSize + i];

    vector<size_t> iotaW(pSize+nSize,0), iotaDirect(pSize+nSize,0);
    for(size_t i = 0, e = pSize+nSize; i != e; i++) iotaW[i] = iotaDirect[i] = i;

    std::sort(iotaW.begin(),iotaW.end(),[&](int a,int b){return (scores[a] > scores[b]) ? true:false;});

    vector<size_t> posW(pSize + nSize,0), posDirect(pSize + nSize,0);
    for(size_t i = 0, e = iotaW.size(); i != e; i++)    posW[iotaW[i]]=i;

    std::sort(iotaDirect.begin(), iotaDirect.begin()+pSize, [&](int a,int b){
        return (scores[a] > scores[b]) ? true:false;
    });

    std::sort(iotaDirect.begin()+pSize, iotaDirect.end(), [&](int a,int b){
        return (scores[a] > scores[b]) ? true:false;
    });

    vector<double> pScores(pSize,0), nScores(nSize,0);
    for(size_t i = 0; i != pSize; i++)  pScores[i] = scores[iotaDirect[i]];
    for(size_t i = 0; i != nSize; i++)  nScores[i] = scores[iotaDirect[i+pSize]];
    vector<size_t> posTmp(pSize+nSize,0);
    DP<positive>(pScores,nScores,posTmp,epsilon);
    for(size_t i = 0, e = posTmp.size(); i != e; i++)
        posDirect[iotaDirect[i]] = posTmp[i];

    ValueType* deriv = new ValueType[rootNodeDeepNet->GetNumEl()];
    std::fill(deriv, deriv + rootNodeDeepNet->GetNumEl(), ValueType(0.0));
    ValueType** diff = DeepNet16->GetRoot()->GetDiffGradientAndEmpMean();
    //cout << "Dimension of gradient: " << rootNodeDeepNet->GetNumEl() << endl;
    ValueType* diffGPU = NULL;
    if (GPUTYPE) {
        cudaMalloc((void**)&diffGPU, sizeof(ValueType)*rootNodeDeepNet->GetNumEl());
        *diff = diffGPU;
    } else {
        *diff = deriv;
    }

    for(int i = 0; i != pSize; i++){
        double sum = 0;
        for(int j = 0; j != nSize; j++){
            int yDirect = (posDirect[i] < posDirect[j + pSize]) ? 1 : -1;
            int yW = (posW[i] < posW[j + pSize]) ? 1 : -1;
            if(positive) sum += yDirect - yW;
            else sum += yW - yDirect;
        }
        sum /= (epsilon * pSize * nSize);
        deriv[i] = sum;
    }

    for(int j = 0; j != nSize; j++){
        double sum = 0;
        for(int i = 0; i != pSize; i++){
            int yDirect = (posDirect[i] < posDirect[j + pSize]) ? 1 : -1;
            int yW = (posW[i] < posW[j + pSize]) ? 1 : -1;
            if(positive) sum -= yDirect - yW;
            else sum -= yW - yDirect;
        }
        sum /= (epsilon * pSize * nSize);
        deriv[pSize + j] = sum;
    }


   // primal += DeepNet16Params.GetRegularization();

    int count = 0;
    double ap = 0;
    for(int i = 0; i < pSize + nSize; i++)
        if(iotaW[i] < pSize){
            count++;
            ap += count / double(i + 1);
        }
    ap /= count;

    if (GPUTYPE) {
        cudaMemcpy(diffGPU, deriv, rootNodeDeepNet->GetNumEl()*sizeof(ValueType), cudaMemcpyHostToDevice);
    }

    DeepNet16->BackwardPass();

    DeepNet16Params.Update(0);

    vector<ValueType> showDerivs;
    DeepNet16Params.GetDerivative(i2t<GPUTYPE>(), showDerivs);
    norm = std::accumulate(showDerivs.begin(), showDerivs.end(), ValueType(0), [](double a, double b){
        return a + b*b;
    });
    norm = std::sqrt(norm);



    DeepNet16Params.ResetGradient(i2t<GPUTYPE>());

    //Reducing step!
    //DeepNet16Params.ReduceStepSize();
    //Storing weights!
//    if (((iter + 1) % 1000) == 0) {
//        std::vector<ValueType>* result = DeepNet16Params.GetWeights(i2t<GPUTYPE>());
//        char tmp[101];
//        sprintf(tmp, "LearntWeights.dat.%d", iter);
//        std::ofstream ofs(tmp, std::ios_base::binary | std::ios_base::out);
//        ofs.write((char*)&(*result)[0], result->size()*sizeof(ValueType));
//        ofs.close();
//    }

    // Modify here later for higher performance
    delete[] deriv;
    if (GPUTYPE) {
        delete[] DeepNet16output;
    }
    return ap;

}

double perceptronTrain(double &norm, CompTree *DeepNet16, ParameterContainer<ValueType, SizeType, GPUTYPE, false> &DeepNet16Params,
                  const int pSize, const int nSize, const double epsilon){

    DeepNet16->ForwardPass(TRAIN);
    ValueType* DeepNet16output;
    NodeType* rootNodeDeepNet = DeepNet16->GetRoot();
    if (GPUTYPE) {
        DeepNet16output = new ValueType[rootNodeDeepNet->GetNumEl()];
        cudaMemcpy((char*)DeepNet16output, rootNodeDeepNet->GetValuePtr(), rootNodeDeepNet->GetNumEl()*sizeof(ValueType), cudaMemcpyDeviceToHost);
    } else {
        DeepNet16output = DeepNet16->GetRoot()->GetValuePtr();
    }

    SizeType* outputDimensions = rootNodeDeepNet->GetSizePtr();
    //assert(outputDimensions[1] == 1000);
    assert(outputDimensions[0] == 1);

    vector<double> scores(pSize+nSize,0);
    for(size_t i = 0; i != pSize; i++)  scores[i] = DeepNet16output[i];
    for(size_t i = 0; i != nSize; i++)  scores[pSize + i] = DeepNet16output[pSize + i];

    //The sum of scores
    double sumScores = 0;
    for(auto e:scores)
        sumScores += e*e;
    std::cout<<"The sum of scores: "<<sumScores<<std::endl;

    vector<size_t> iotaW(pSize+nSize,0);
    for(size_t i = 0, e = pSize+nSize; i != e; i++) iotaW[i] = i;

    std::sort(iotaW.begin(),iotaW.end(),[&](int a,int b){return (scores[a] > scores[b]) ? true:false;});

    vector<size_t> posW(pSize + nSize,0), pos(pSize + nSize,0);
    for(size_t i = 0, e = iotaW.size(); i != e; i++)    posW[iotaW[i]]=pos[i]=i;

    ValueType* deriv = new ValueType[rootNodeDeepNet->GetNumEl()];
    std::fill(deriv, deriv + rootNodeDeepNet->GetNumEl(), ValueType(0.0));
    ValueType** diff = DeepNet16->GetRoot()->GetDiffGradientAndEmpMean();
    ValueType* diffGPU = NULL;
    if (GPUTYPE) {
        cudaMalloc((void**)&diffGPU, sizeof(ValueType)*rootNodeDeepNet->GetNumEl());
        *diff = diffGPU;
    } else {
        *diff = deriv;
    }

    for(int i = 0; i != pSize; i++){
        double sum = 0;
        for(int j = 0; j != nSize; j++){
            int y = (pos[i] < pos[j + pSize]) ? 1 : -1;
            int yW = (posW[i] < posW[j + pSize]) ? 1 : -1;
            sum += yW - y;
        }
        sum /= (epsilon * pSize * nSize);
        deriv[i] = sum;
    }

    for(int j = 0; j != nSize; j++){
        double sum = 0;
        for(int i = 0; i != pSize; i++){
            int y = (pos[i] < pos[j + pSize]) ? 1 : -1;
            int yW = (posW[i] < posW[j + pSize]) ? 1 : -1;
            sum -= yW - y;
       }
        sum /= (epsilon * pSize * nSize);
        deriv[j + pSize] = sum;
    }

    // What's the deriv in loss layer?
    double sumderiv = 0;
    for(int i = 0;i != pSize + nSize; i++)
        sumderiv += deriv[i]*deriv[i];
    std::cout<<"The norm of deriv at top: "<<sumderiv<<std::endl;
    std::cout<<"The regularizer: "<<DeepNet16Params.GetRegularization()<<std::endl;
    int count = 0;
    double ap = 0;
    for(int i = 0; i < pSize + nSize; i++)
        if(iotaW[i] < pSize){
            count++;
            ap += count / double(i + 1);
        }
    ap /= count;

    if (GPUTYPE) {
        cudaMemcpy(diffGPU, deriv, rootNodeDeepNet->GetNumEl()*sizeof(ValueType), cudaMemcpyHostToDevice);
    }

    DeepNet16->BackwardPass();
    DeepNet16Params.Update(0);
    vector<ValueType> showDerivs;
    DeepNet16Params.GetDerivative(i2t<GPUTYPE>(), showDerivs);
    norm = std::sqrt(std::accumulate(showDerivs.begin(), showDerivs.end(), ValueType(0), [&](double a, double b){
        return a + b*b;
    }));

    DeepNet16Params.ResetGradient(i2t<GPUTYPE>());

    //Reducing step!
    //DeepNet16Params.ReduceStepSize();
    //Storing weights!
//    if (((iter + 1) % 1000) == 0) {
//        std::vector<ValueType>* result = DeepNet16Params.GetWeights(i2t<GPUTYPE>());
//        char tmp[101];
//        sprintf(tmp, "LearntWeights.dat.%d", iter);
//        std::ofstream ofs(tmp, std::ios_base::binary | std::ios_base::out);
//        ofs.write((char*)&(*result)[0], result->size()*sizeof(ValueType));
//        ofs.close();
//    }
    // Modify here later for higher performance

    delete[] deriv;
    if (GPUTYPE) {
        delete[] DeepNet16output;
    }
    return ap;

}


double APSVMTrain(double &norm, CompTree *DeepNet16, ParameterContainer<ValueType, SizeType, GPUTYPE, false> &DeepNet16Params,
                  const int pSize, const int nSize){
    const double epsilon = 1.0;
    DeepNet16->ForwardPass(TRAIN);
    ValueType* DeepNet16output;
    NodeType* rootNodeDeepNet = DeepNet16->GetRoot();
    if (GPUTYPE) {
        DeepNet16output = new ValueType[rootNodeDeepNet->GetNumEl()];
        cudaMemcpy((char*)DeepNet16output, rootNodeDeepNet->GetValuePtr(), rootNodeDeepNet->GetNumEl()*sizeof(ValueType), cudaMemcpyDeviceToHost);
    } else {
        DeepNet16output = DeepNet16->GetRoot()->GetValuePtr();
    }

    SizeType* outputDimensions = rootNodeDeepNet->GetSizePtr();
//    assert(outputDimensions[1] == 2);
    assert(outputDimensions[0] == 1);

    vector<double> scores(pSize+nSize,0);
    for(size_t i = 0; i != pSize; i++)  scores[i] = DeepNet16output[i];
    for(size_t i = 0; i != nSize; i++)  scores[pSize + i] = DeepNet16output[pSize + i];

    vector<size_t> iotaDirect(pSize+nSize,0);
    for(size_t i = 0, e = pSize+nSize; i != e; i++) iotaDirect[i] = i;

    vector<size_t> pos(pSize + nSize,0), posDirect(pSize + nSize,0);
    for(size_t i = 0, e = pos.size(); i != e; i++)    pos[i]=i;

    std::sort(iotaDirect.begin(), iotaDirect.begin()+pSize, [&](int a,int b){
        return (scores[a] > scores[b]) ? true:false;
    });

    std::sort(iotaDirect.begin()+pSize, iotaDirect.end(), [&](int a,int b){
        return (scores[a] > scores[b]) ? true:false;
    });

    vector<double> pScores(pSize,0), nScores(nSize,0);
    for(size_t i = 0; i != pSize; i++)  pScores[i] = scores[iotaDirect[i]];
    for(size_t i = 0; i != nSize; i++)  nScores[i] = scores[iotaDirect[i+pSize]];
    vector<size_t> posTmp(pSize+nSize,0);
    DP<true>(pScores,nScores,posTmp,epsilon);
    for(size_t i = 0, e = posTmp.size(); i != e; i++)
        posDirect[iotaDirect[i]] = posTmp[i];

    ValueType* deriv = new ValueType[rootNodeDeepNet->GetNumEl()];
    std::fill(deriv, deriv + rootNodeDeepNet->GetNumEl(), ValueType(0.0));
    ValueType** diff = DeepNet16->GetRoot()->GetDiffGradientAndEmpMean();
    //cout << "Dimension of gradient: " << rootNodeDeepNet->GetNumEl() << endl;
    ValueType* diffGPU = NULL;
    if (GPUTYPE) {
        cudaMalloc((void**)&diffGPU, sizeof(ValueType)*rootNodeDeepNet->GetNumEl());
        *diff = diffGPU;
    } else {
        *diff = deriv;
    }

    for(int i = 0; i != pSize; i++){
        double sum = 0;
        for(int j = 0; j != nSize; j++){
            int yDirect = (posDirect[i] < posDirect[j + pSize]) ? 1 : -1;
            int y = (pos[i] < pos[j + pSize]) ? 1 : -1;
            sum += yDirect - y;
        }
        sum /= (epsilon * pSize * nSize);
        deriv[i] = sum;
    }

    for(int j = 0; j != nSize; j++){
        double sum = 0;
        for(int i = 0; i != pSize; i++){
            int yDirect = (posDirect[i] < posDirect[j + pSize]) ? 1 : -1;
            int y = (pos[i] < pos[j + pSize]) ? 1 : -1;
            sum -= yDirect - y;
        }
        sum /= (epsilon * pSize * nSize);
        deriv[pSize + j] = sum;
    }

   // primal += DeepNet16Params.GetRegularization();
    vector<size_t> iotaW(pSize+nSize,0);
    for(size_t i = 0, e = pSize+nSize; i != e; i++) iotaW[i] = i;

    std::sort(iotaW.begin(),iotaW.end(),[&](int a,int b){return (scores[a] > scores[b]) ? true:false;});
    int count = 0;
    double ap = 0;
    for(int i = 0; i < pSize + nSize; i++)
        if(iotaW[i] < pSize){
            count++;
            ap += count / double(i + 1);
        }
    ap /= count;

    if (GPUTYPE) {
        cudaMemcpy(diffGPU, deriv, rootNodeDeepNet->GetNumEl()*sizeof(ValueType), cudaMemcpyHostToDevice);
    }

    DeepNet16->BackwardPass();
    DeepNet16Params.Update(0);

    vector<ValueType> showDerivs;
    DeepNet16Params.GetDerivative(i2t<GPUTYPE>(), showDerivs);
    norm = std::sqrt(std::accumulate(showDerivs.begin(), showDerivs.end(), ValueType(0), [&](double a, double b){
        return a + b*b;
    }));
    double signsum = std::accumulate(showDerivs.begin(),showDerivs.end(),ValueType(0),[&](double a,double b){
        return a+b;
    });
    std::cout <<"The sum of derivs: "<<signsum<<std::endl;

    double w2 = DeepNet16Params.GetRegularization();
    std::cout <<"The regularizer: "<<w2 <<std::endl;

    double w2 = DeepNet16Params.GetRegularization();
    std::cout <<"Hey, this is the regularizer: "<<w2 <<std::endl;

    DeepNet16Params.ResetGradient(i2t<GPUTYPE>());

    //Reducing step!
    //DeepNet16Params.ReduceStepSize();
    //Storing weights!
//    if (((iter + 1) % 1000) == 0) {
//        std::vector<ValueType>* result = DeepNet16Params.GetWeights(i2t<GPUTYPE>());
//        char tmp[101];
//        sprintf(tmp, "LearntWeights.dat.%d", iter);
//        std::ofstream ofs(tmp, std::ios_base::binary | std::ios_base::out);
//        ofs.write((char*)&(*result)[0], result->size()*sizeof(ValueType));
//        ofs.close();
//    }
    // Modify here later for higher performance

    delete[] deriv;
    if (GPUTYPE) {
        delete[] DeepNet16output;
    }
    return ap;
}

void genData(string name, int D , int N, bool picture, const int GPUid){
    if (GPUTYPE) {
        int GPUboard = GPUid;
        auto tmp = cudaSetDevice(GPUboard);
        std::cerr << tmp << std::endl;
        if(tmp != cudaSuccess) {
            std::cout << "Cannot set GPU device " << GPUboard << std::endl;
            return;
        } else {
            std::cout << "Using GPU " << GPUboard << std::endl;
        }
    } else {
        std::cout << "NOT using GPU. Are you sure?" << std::endl;
    }
    CompTree *dnn = new CompTree;
    ParameterContainer<ValueType,SizeType,GPUTYPE,false> paramContainer;
    DataType *data = new DataType(DataType::NodeParameters());

    std::random_device rd;
    std::mt19937 rg(rd());
    std::normal_distribution<ValueType> dist(0,10);

    ValueType *rnumbers = new ValueType[D*N]();
    for(int i = 0; i < N; i++)
        for(int j = 0; j < D; j++)
            rnumbers[i * D + j] = dist(rg);

    ValueType *dataPtrGPU = NULL;
    if (GPUTYPE) {
        cudaMalloc((void**)&dataPtrGPU, sizeof(ValueType)*D*N);
        cudaMemcpy((char*)dataPtrGPU, (char*)rnumbers, sizeof(ValueType)*D*N, cudaMemcpyHostToDevice);
        data->SetValueSize(dataPtrGPU, new SizeType[2]{D,N}, 2);
    } else {
        data->SetValueSize(rnumbers, new SizeType[2]{D,N}, 2);
    }
    createTestNet(1,1,dnn,paramContainer,data);

    // Setting the ground truth!
    vector<ValueType> initWeights(paramContainer.GetWeightDimension(),ValueType(1.0));
    for(auto &v: initWeights)   v = dist(rg);
    paramContainer.SetWeights(i2t<GPUTYPE>(), &initWeights);

    dnn->ForwardPass(TRAIN);

    ValueType * netOutput;
    NodeType* rootNodeDeepNet = dnn->GetRoot();
    if (GPUTYPE) {
        netOutput = new ValueType[rootNodeDeepNet->GetNumEl()];
        cudaMemcpy((char*)netOutput, rootNodeDeepNet->GetValuePtr(), rootNodeDeepNet->GetNumEl()*sizeof(ValueType), cudaMemcpyDeviceToHost);
    } else {
        netOutput = dnn->GetRoot()->GetValuePtr();
    }

    SizeType* outputDim = dnn->GetRoot()->GetSizePtr();
    assert(outputDim[0] == 1);
    assert(outputDim[1] == N);
//    cout<<"-----original generated scores----\n";
//    for(int i = 0;i < outputDim[1];i++)
//        cout << netOutput[i] <<endl;
//
    vector<double> scores;
    std::copy(netOutput, netOutput+outputDim[1],std::back_inserter(scores));
    vector<int> iota(outputDim[1],0);
    std::iota(iota.begin(),iota.end(),0);
    std::sort(iota.begin(),iota.end(),[&](int a,int b){
        return (scores[a] > scores[b]) ? true:false;
    });

//    cout<<"-----sorted generated scores-----\n";
//    for(auto v:iota)
//        cout<<scores[v]<<endl;

    int pSize = int(outputDim[1] * 0.2);
    int nSize = outputDim[1] - pSize;

    std::ofstream fout(name);
    using std::endl;
    fout.precision(30);
    fout<<1<<endl;
    fout<<pSize<<" "<<D<<endl;

//    ValueType *newnumbers = new ValueType[D * N]();
    for(int i = 0; i < pSize; i++){
        for(int k = 0;k < D; k++)
        {
            fout << rnumbers[iota[i] * D + k] << " ";
 //           newnumbers[i * D + k] = rnumbers[iota[i] * D + k];
        }
        fout<<endl;
    }

    fout<<nSize<<" "<<D<<endl;
    for(int i = 0; i < nSize; i++){
        for(int k = 0;k < D;k++){
            fout << rnumbers[iota[i + pSize] * D + k] << " ";
  //          newnumbers[(i + pSize) * D + k] = rnumbers[iota[i + pSize]*D + k];
        }
        fout<<endl;
    }

   // data->SetValueSize(newnumbers, new SizeType[2]{D, N}, 2);
    //dnn->ForwardPass(TRAIN);

//    netOutput = dnn->GetRoot()->GetValuePtr();
//    cout<<"-----reconstructed values------"<<endl;
//    for(int i = 0; i < N; i++)
//        cout << netOutput[i] << endl;
//    cout<<"-------------------------------\n";
//
//    data->SetValueSize(rnumbers, new SizeType[2]{D, N}, 2);
//    dnn->ForwardPass(TRAIN);
//
//    netOutput = dnn->GetRoot()->GetValuePtr();
//    cout<<"-----reconstructed values 2------"<<endl;
//    for(int i = 0; i < N; i++)
//        cout << netOutput[i] << endl;
//    cout<<"-------------------------------\n";
//
    fout.close();
    if(picture){
        std::ofstream pic("picture.txt");
        pic.precision(10);
        for(double i = -5; i < 5; i += 0.1){
            for(double j = -5; j < 5; j += 0.1){
                rnumbers[0] = i;
                rnumbers[1] = j;
                dnn->ForwardPass(TRAIN);
 //               cout << netOutput[0] << " ";
                pic << netOutput[0] << " ";
            }
            pic << std::endl;
//            cout << std::endl;
        }
        pic.close();
    }

    deleteCNN(dnn,paramContainer);
}

template double performTrain<true>(double &norm, CompTree *DeepNet16, ParameterContainer<ValueType, SizeType, GPUTYPE, false> &DeepNet16Params,
                  const int pSize, const int nSize, const double epsilon);
template double performTrain<false>(double &norm, CompTree *DeepNet16, ParameterContainer<ValueType, SizeType, GPUTYPE, false> &DeepNet16Params,
                  const int pSize, const int nSize, const double epsilon);


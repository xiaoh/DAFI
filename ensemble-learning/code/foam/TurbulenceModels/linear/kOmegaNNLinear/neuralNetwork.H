
//********* Load case-specific neural network
// inputs
// TODO: these might change. Should be inputs in turbulenceProperties subdictionary with defaults. 
const char* savedModelDir = "model/";
const char* tag = "serve";
const char* inputTensorName = "serving_default_cpp_norm_input";
const char* outputTensorName = "StatefulPartitionedCall";

// read the model
TF_Graph* Graph = TF_NewGraph();
TF_Status* Status = TF_NewStatus();
TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
TF_Buffer* RunOpts = NULL;
int ntags = 1;
TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, savedModelDir, &tag, ntags, Graph, NULL, Status);

// get input tensor
int NumInputs = 1;
TF_Output* Input = static_cast<TF_Output*>(malloc(sizeof(TF_Output) * NumInputs));
TF_Output t0 = {TF_GraphOperationByName(Graph, inputTensorName), 0};
Input[0] = t0;
TF_Tensor** InputValues = static_cast<TF_Tensor**>(malloc(sizeof(TF_Tensor*)*NumInputs));

// get output tensor
int NumOutputs = 1;
TF_Output* Output = static_cast<TF_Output*>(malloc(sizeof(TF_Output) * NumOutputs));
TF_Output t2 = {TF_GraphOperationByName(Graph, outputTensorName), 0};
if(t2.oper == NULL)
    printf("ERROR: Failed TF_GraphOperationByName %s\n", outputTensorName);
else	
printf("TF_GraphOperationByName %s is OK\n", outputTensorName);
Output[0] = t2;
TF_Tensor** OutputValues = static_cast<TF_Tensor**>(malloc(sizeof(TF_Tensor*)*NumOutputs));


//********* Set input data values
int ndims = 2; 
int ndata = sizeof(double)*nCells*nScalarInvariants; 
int64_t dims[] = {nCells,nScalarInvariants}; 
double data[nCells*nScalarInvariants];
for(int i=0;i< (nCells*nScalarInvariants); i++)
{
    data[i] = theta1_[i];
}
TF_Tensor* int_tensor = TF_NewTensor(TF_DOUBLE, dims, ndims, data, ndata, &NoOpDeallocator, 0);
InputValues[0] = int_tensor;


//********* Run session
TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);
// free memory
TF_DeleteGraph(Graph);
TF_DeleteSession(Session, Status);
TF_DeleteSessionOptions(SessionOpts);
TF_DeleteStatus(Status);


//********* Read output data values
void* buff = TF_TensorData(OutputValues[0]);
double* g1nn = static_cast<double*>(buff);


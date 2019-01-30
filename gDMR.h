#ifdef __cplusplus
extern "C" {
#endif
// Initialize the solver. 
// blocks and threadsPerBlock: GPU computation settings (see cudaToolkit documentation).
// devList: CPU recognized integers identifying all GPU devices the solver will use.
// numDevs: length of devList.
// elements: length of a vector in the linear system.
// basisDim: dimensionality of total solution basis (Krylov + deflation), basisDim + 1 must be
// divisible by the number of devices.
// deflatDim: dimensionality of the deflation basis, must be divisible by the number of devices.
// linComp: linear operator function.
void initDMR(int blocks, int threadsPerBlock, int *devList, int numDevs, int elements, int basisDim, int deflatDim, void(*linComp)(void));
// Free solver memory.
void freeDMR(void);
// Inverse solver, returns norm of residual.
// mode: a value of !0 indicates that the solver has been previously called on a similar linear 
// system and that this existing deflation space should be used in the first Arnoldi iteration.
// extSrcVec: source vector (right hand side) that will be acted on by the linear 
// operator.
// extImgVec: image vector (left hand side) holding the results from the linear operator.
// extStream: stream identifier for synchronizing the linear operator.
// extDevNum: device identifier for interaction with the linear operator.
// solTol: Relative allowable magnitude of the residual, difference between true and approximate
// images.
// matRank: rank of the linear operator, used to determine what initial vector should
// be used to begin a new Arnoldi iteration. 
// numIts: stores the number of iterations used in the inverse solve.
float solverDMR(int mode, cuComplex *extSrcVec, cuComplex *extImgVec, cudaStream_t extStream, int extDevNum, float solTol, int matRank, int *numIts);
// Import and export source side singular vectors into / out of the solver. 
// The size of CPU memory location must be consistent with the latest initialization, ie. 
// deflatDim * elements * sizeof(float _Complex)
void impSVDMR(float _Complex *cMem);
void expSVDMR(float _Complex *cMem);
#ifdef __cplusplus
}
#endif
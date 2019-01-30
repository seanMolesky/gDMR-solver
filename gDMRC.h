#ifdef __cplusplus
extern "C" {
#endif
// Initialize the linear system. basisSizeMDR + 1 and deflatSizeMDR must be divisible
 // by the number of devices used by the iterative solver, numDevsMDR.
void initLS(int blocksMDR, int threadsPerBlockMDR, int devNumLO, int cells, int *devListMDR, int numDevsMDR, int basisSizeMDR, int deflatSizeMDR);
// Free the linear system.
void freeLS(void);
// Import / Export image vector to / from device, the right hand side y in a linear system A x = y.
void impImgLS(float _Complex *cImg);
void expImgLS(float _Complex *cImg);
// Import / Export source vector to / from device, the solution guess x in the linear system A x = y.
void impSrcLS(float _Complex *cSrc);
void expSrcLS(float _Complex *cSrc);
// Import /Export values for the linear operator, the matrix A in the linear system A x = y. 
void impOptLS(float _Complex *cOpt);
void expOptLS(float _Complex *cOpt);
// Solve linear system using gMDR, returning the residual norm of the solution.
// A mode value of !0 indicates that the solver has been previously called on a similar linear 
// system, and should use the existing deflation space in its first iteration.
// numIts stores the number of iterations required by the inverse solver.
float solveLS(int mode, float solTol, int matRank, int *numIts);
// Perform the linear operation, updating the source vector.
void linOptLS(void);
// Import and export source side singular vectors into / out of the solver. 
// The size of CPU memory location must be consistent with the latest initialization, ie. 
// deflatDim * elements * sizeof(float _Complex)
void impSVDMR(float _Complex *cMem);
void expSVDMR(float _Complex *cMem);
#ifdef __cplusplus
}
#endif
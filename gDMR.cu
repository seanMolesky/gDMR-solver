#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "gDMR.h"
// Arrays and coefficient vectors referencing the Krylov / deflation basis vectors. 
// hessArrDMR: Hessenberg coefficient array.
// hessSrcDMR: coefficients before multiplication with the Hessenberg matrix, 
// RHS, source side basis. 
// hessTrgDMR: coefficients after multiplication with the Hessenberg matrix,
// LHS, target side basis. 
// hessPrjDMR: projection coefficients / goal coefficients for a given iteration.
// hessWrkDMR: work space vector for reordering coefficient when multiple devices are 
// present.
cuComplex *hessArrDMR;
cuComplex *hessSrcDMR;
cuComplex *hessTrgDMR;
cuComplex *hessWrkDMR;
cuComplex *hessPrjDMR;
// Memory for QR decomposition of the Hessenberg matrix.
int wSizeQR;
cuComplex *qHessArrDMR;
cuComplex *rHessArrDMR;
// CUSOLVER arrays
cuComplex *tCUSLVDMR;
cuComplex *cpyHessArrDMR;
cuComplex *wCUSLVDMR;
cusolverDnHandle_t slvHandleQRDMR;
cudaStream_t slvStreamQRDMR;
// Memory for SV decompositions.
// uHessArrDMR: target side singular vectors.
// eHessArrDMR: singular values.
// vHessArrDMR: source side.
cuComplex *uHessArrDMR;
float *eHessArrDMR;
cuComplex *vHessArrDMR;
cuComplex *svWorkDMR;
gesvdjInfo_t geSVjPrmsDMR;
cusolverDnHandle_t slvHandleSVDMR;
cudaStream_t slvStreamSVDMR;
int workSizeSVDMR;
// Additional memory for ``head" (master) device.
// genWorkVecDMR: general work space vector.
// sumWorkVecDMR: work space for partial sums.
// resVecDMR: image residual, persists across iterations.
// aSrcVecDMR: approximate source.
cuComplex *genWorkVecDMR;
cuComplex *sumWorkVecDMR;
cuComplex *resVecDMR;
cuComplex *aSrcVecDMR;
// Per device memory for generating vectors from the Krylov and deflation bases.
cuComplex **kryCoeffsDMR;
cuComplex **defCoeffsDMR;
// Per device memory holding basis vectors, and source side basis vectors for the deflation space.
cuComplex **itrBasisDMR;
cuComplex **defBasisDMR;
// Accompanying work areas.
cuComplex **itrVecDMR;
cuComplex **defWorkDMR;
// Inner product results.
cuComplex *innProdDevDMR;
// CUDA handles.
cublasHandle_t *blasHandleDMR;
cudaStream_t *blasStreamDMR;
cublasStatus_t blasStatusDMR = CUBLAS_STATUS_SUCCESS;
cusolverStatus_t solverStatusDMR = CUSOLVER_STATUS_SUCCESS;
// Global settings for GPUs
int *devListDMR;
int numDevsDMR;
int blocksDMR;
int threadsPerBlockDMR;
// Per device size of the dimension of total and deflation bases.
int localBasDimDMR;
int localDefDimDMR;
// Dimensionality of the total basis and deflation basis.
int basisDimDMR;
int deflateDimDMR;
// Number of elements in a solution vector.
int numElementsDMR;
size_t vecSizeDMR;
size_t hessenbergSizeDMR;
// SV decomposition settings
const cusolverEigMode_t vecModeSVDMR = CUSOLVER_EIG_MODE_VECTOR;
const int memModeSVDMR = 0;
const double svTolDMR = 2.e-7;
const int maxSVSweepsDMR = 128;
// Memory for function pointer providing linear iteration (matrix multiplication).
void (*linOptDMR)(void);
// Flag signifying that gDMR is stuck in a loop, and should move to set a different set
// of basis vectors.
int kryLoopFlagDMR;
int kryFlagCountDMR;
int kryBasisLoopsDMR;

__global__
void zeroArrayDMR(int startInd, int endInd, cuComplex *array)
{
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for(int i = localId + startInd; i < endInd; i += stride)
	{
		array[i].x = 0.0;
		array[i].y = 0.0;
	}

	return;		
}
// Reorder Hessenberg coefficient vector for consistency with multi device storage conventions.
// Mode -1 changes non-standard order to standard, all other integers standard to non-standard.
__global__
void reorderHessCoeffs(int mode, int numDevs, int startPos, int vecSize, int localBasisDim, const cuComplex *orgnVec, cuComplex *rordVec)
{
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;
	int devOffset, memOffset;
	int rordPos;
	int j;

	if(mode != -1)
	{
		for(int i = localId; i < vecSize ; i += stride)
		{
			if(i < startPos)
			{
				rordVec[i] = orgnVec[i];
			}
			else
			{	
				j = i - startPos;
				devOffset = j % numDevs;
				memOffset = j / numDevs;
				rordPos = devOffset * localBasisDim + memOffset + startPos;
				rordVec[rordPos] = orgnVec[i];
			}
		}
	}
	else
	{
		for(int i = localId; i < vecSize ; i += stride)
		{
			if(i < startPos)
			{
				rordVec[i] = orgnVec[i];
			}
			else
			{
				j =  i - startPos;
				devOffset = j % numDevs;
				memOffset = j / numDevs;
				rordPos = devOffset * localBasisDim + memOffset + startPos;
				rordVec[i] = orgnVec[rordPos];
			}
		}
	}

	return;
}

__global__
void eyeArrayDMR(int startInd, int rows, int cols, cuComplex *array)
{
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;
	int colNum, rowNum;

	for(int i = localId + startInd; i < rows * cols; i += stride)
	{
		rowNum = i % rows;
		colNum = i / rows;

		if(rowNum == colNum)
		{
			array[i].x = 1.0;
			array[i].y = 0.0;
		}
	}

	return;	
}

__global__
void vecAddDMR(int mode, int numElements, cuComplex *sumVec, const cuComplex *updateVec)
{
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	if(mode == -1)
	{
		for(int i = localId; i < numElements; i += stride)
		{
			sumVec[i].x -= updateVec[i].x;
			sumVec[i].y -= updateVec[i].y;
		}
	}
	else
	{
		for(int i = localId; i < numElements; i += stride)
		{
			sumVec[i].x += updateVec[i].x;
			sumVec[i].y += updateVec[i].y;
		}	
	}

	return;		
}

__global__
void vecAddArrDMR(int arrayDim, int numElements, cuComplex *sumVec, const cuComplex *updateArray)
{
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for(int i = localId; i < numElements; i += stride)
	{
		for(int j = 0; j < arrayDim; j++)
		{
			sumVec[i].x += updateArray[i + j * numElements].x;
			sumVec[i].y += updateArray[i + j * numElements].y;
		}
	}

	return;
}

__global__
void vecScaleDiffDMR(int numCoeffs, int numCells, const cuComplex *coeffs, const cuComplex *vecBasis, float scale, cuComplex *mutateVec)
{
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for(int i = localId; i < numCells; i += stride)
	{	
		mutateVec[i].x = mutateVec[i].x * scale;
		mutateVec[i].y = mutateVec[i].y * scale;

		for(int j = 0; j < numCoeffs; j++)
		{
			mutateVec[i].x -= coeffs[j].x * vecBasis[j * numCells + i].x - coeffs[j].y * vecBasis[j * numCells + i].y;
			mutateVec[i].y -= coeffs[j].x * vecBasis[j * numCells + i].y + coeffs[j].y * vecBasis[j * numCells + i].x;
		}
	}

	return;
}

__global__
void addBasisVecsDMR(int numCoeffs, int numCells, const cuComplex *coeffs, const cuComplex *vecBasis, cuComplex *mutateVec)
{
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for(int i = localId; i < numCells; i += stride)
	{	
		for(int j = 0; j < numCoeffs; j++)
		{
			mutateVec[i].x += coeffs[j].x * vecBasis[j * numCells + i].x - coeffs[j].y * vecBasis[j * numCells + i].y;
			mutateVec[i].y += coeffs[j].x * vecBasis[j * numCells + i].y + coeffs[j].y * vecBasis[j * numCells + i].x;
		}
	}

	return;
}

__global__
void innerProdKerDMR(int numElements, int blocks, const cuComplex *linFunc, const cuComplex *vec, cuComplex *blockProds, cuComplex *prod)
{	
	extern __shared__ cuComplex lCache[];

	cuComplex threadSum;
	threadSum.x = 0.0;
	threadSum.y = 0.0;
	int lId = threadIdx.x;
	int gId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;
	int rLevel = blockDim.x / 2;

	for(int i = gId; i < numElements; i += stride)
	{
		threadSum.x += linFunc[i].x * vec[i].x + linFunc[i].y * vec[i].y;
		threadSum.y += linFunc[i].x * vec[i].y - linFunc[i].y * vec[i].x;
	}

	lCache[lId].x = threadSum.x;
	lCache[lId].y = threadSum.y;
	__syncthreads();

	while (rLevel != 0)
	{
		if(lId < rLevel)
		{
			lCache[lId].x += lCache[lId+rLevel].x;
			lCache[lId].y += lCache[lId+rLevel].y;
		}
		__syncthreads();
		rLevel /= 2;
	}

	if(lId == 0)
	{
		blockProds[blockIdx.x].x = lCache[0].x;
		blockProds[blockIdx.x].y = lCache[0].y;
	}
	
	__syncthreads();

	if(gId == 0)
	{
		prod[0].x = 0.0;
		prod[0].y = 0.0;

		for(int i = 0; i < blocks; i++)
		{
			prod[0].x += blockProds[i].x;
			prod[0].y += blockProds[i].y;
		}

		prod[0].x = prod[0].x;
		prod[0].y = prod[0].y;
	}

	return;	
}

__global__ 
void normalizeVecDMR(int mode, int numCells, cuComplex *vec, cuComplex *norm, float nNorm)
{ 
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;
	float realNorm;

	if(mode == -1)
	{
		realNorm = 1.0 / sqrtf(norm[0].x);
	}
	else if(mode == 2)
	{
		realNorm = nNorm;
	}
	else
	{
		realNorm = sqrtf(norm[0].x);
	}
	
	for(int i = localId; i < numCells; i += stride)
	{
		vec[i].x = vec[i].x * realNorm;
		vec[i].y = vec[i].y * realNorm;
	}

	return;	
}
// Function for examining device data. Otherwise unessential.
extern "C"{__host__
	void gPrintDMR(int rows, int cols, cuComplex *gArray, int devNum)
	{
		int rowNum;
		int colNum;

		if(cudaSetDevice(devNum) != cudaSuccess)
		{
			fprintf(stderr, "gPrintDMR CUDA Error: Failed to switch to device %d.\n", devNum);
			return;
		}

		if(cudaDeviceSynchronize() != cudaSuccess)
		{
			fprintf(stderr, "gPrintDMR CUDA Error: Failed to synchronize device %d.\n", devNum);
			return;	
		}

		cuComplex *hostMem;
		hostMem = (cuComplex*)malloc(sizeof(cuComplex) * rows * cols);

		if(cudaMemcpy(hostMem, gArray, sizeof(cuComplex) * rows * cols, cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			fprintf(stderr, "gPrintDMR CUDA Error: Failed to copy data to host.\n");
			return;	
		}	

		fprintf(stdout, "\n");

		for(int itr = 0; itr < rows * cols; itr++)
		{
			colNum = itr % cols;
			rowNum = itr / cols;

			if((itr + 1) % cols == 0)
			{
				fprintf(stdout, "%4.3f+%4.3fi\n", hostMem[rowNum + colNum * rows].x, hostMem[rowNum + colNum * rows].y);
			}
			else
			{
				fprintf(stdout, "%4.3f+%4.3fi ", hostMem[rowNum + colNum * rows].x, hostMem[rowNum + colNum * rows].y);
			}
		}

		fprintf(stdout, "\n");
		free(hostMem);
		return;
	}
}
// Resets all device memory, called if an error is detected. Note that this function will 
// produce segmentation faults if called for multiple virtual devices residing on the same 
// physical device.
extern "C"{__host__
	void fullResetDMR(int devUpperBound)
	{
		for(int devItr = 0; devItr < devUpperBound; devItr++)
		{
			if(cudaSetDevice(devListDMR[devItr]) != cudaSuccess)
			{
				fprintf(stderr, "fullResetDMR CUDA Error: Failed to switch to device %d.\n", devItr);
				return;
			}			

			if(cudaDeviceReset() != cudaSuccess)
			{
				fprintf(stderr, "fullResetDMR CUDA Error: Failed to reset device %d.\n", devItr);
				return;
			}
		}		

		return;
	}
}

extern "C"{__host__
	void impSVDMR(float _Complex *cMem)
	{
		for(int devItr = 0; devItr < numDevsDMR; devItr++)
		{	
			if (cudaSetDevice(devItr) != cudaSuccess)
			{
				fprintf(stderr, "impSVDMR CUDA Error: Failed to set device.\n");
				resetLS();
				return;
			}
			
			if(cudaMemcpy(defBasisDMR[devItr], &(cMem[devItr * numElementsDMR * localDefDim]), sizeof(cuComplex) * numElementsDMR * localDefDim, cudaMemcpyHostToDevice) != cudaSuccess)
			{
				fprintf(stderr, "impSVDMR CUDA Error: Failed to import singular vectors on device %d.\n", devItr);
				resetLS();
				return;
			}
		}

		return;
	}
}

extern "C"{__host__
	void expSVDMR(float _Complex *cMem)
	{
		for(int devItr = 0; devItr < numDevsDMR; devItr++)
		{	
			if (cudaSetDevice(devItr) != cudaSuccess)
			{
				fprintf(stderr, "impSVDMR CUDA Error: Failed to set device.\n");
				resetLS();
				return;
			}
			
			if(cudaMemcpy(&(cMem[devItr * numElementsDMR * localDefDim]), defBasisDMR[devItr], sizeof(cuComplex) * numElementsDMR * localDefDim, cudaMemcpyHostToDevice) != cudaSuccess)
			{
				fprintf(stderr, "impSVDMR CUDA Error: Failed to export singular vectors from device %d.\n", devItr);
				resetLS();
				return;
			}
		}

		return;
	}
}

extern "C"{__host__
	int isPowerTwo(int n) 
	{ 
		if(fmod(log2((float) n), 1.0) < 0.01)
		{
			return 0;
		}
		else
		{
			return 1;
		} 
	}
}

extern "C"{__host__
	int innerProdDMR(int numElements, const cuComplex *linFunc, const cuComplex *vec, cuComplex *prod, int devNum)
	{
		cudaStream_t prodStream;
		cuComplex *blockProds;

		if(cudaSetDevice(devNum) != cudaSuccess)
		{
			fprintf(stderr, "innerProdDMR CUDA Error: Failed to switch to device %d.\n", devNum);
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(isPowerTwo(blocksDMR) != 0)
		{
			fprintf(stderr, "innerProdDMR Error: Reduction algorithm requires the number of blocksDMR, %d, to be a power of 2.\n", blocksDMR);
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(cudaMalloc((void**) &blockProds, blocksDMR * sizeof(cuComplex)) != cudaSuccess) 
		{
			fprintf(stderr, "innerProdDMR CUDA Error: Failed to allocate device memory for product reduction.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(cudaStreamCreate(&prodStream) != cudaSuccess)
		{
			fprintf(stderr, "innerProdDMR CUDA Error: Failed to initialize stream for inner product.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		innerProdKerDMR<<<blocksDMR, threadsPerBlockDMR, threadsPerBlockDMR * sizeof(cuComplex), prodStream>>>(numElements, blocksDMR, linFunc, vec, blockProds, prod);

		if(cudaStreamSynchronize(prodStream) != cudaSuccess) 
		{
			fprintf(stderr, "innerProdDMR CUDA Error: Failed to synchronize inner product stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		cudaFree(blockProds);

		if(cudaStreamDestroy(prodStream) != cudaSuccess)
		{
			fprintf(stderr, "innerProdDMR CUDA Error: Failed to destroy inner product stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		return 0;
	}
}
// Computes projection coefficients of a source vector against the column vectors of 
// a matrix, operationMat. The result is stored in target.
extern "C"{__host__
	int basisProjDMR(const cuComplex* operationMat, const cuComplex* source, cuComplex* target, int basisDim, int numElements, int devItr)
	{
		cuComplex alpha, beta;
		alpha.x = 1.0;
		alpha.y = 0.0;
		beta.x = 0.0;
		beta.y = 0.0;

		if(cudaSetDevice(devListDMR[devItr]) != cudaSuccess)
		{
			fprintf(stderr, "basisProjDMR CUDA Error: Failed to switch to device %d.\n", devListDMR[devItr]);
			fullResetDMR(numDevsDMR);
			return 1;
		}

		blasStatusDMR = cublasCgemv(blasHandleDMR[devItr], CUBLAS_OP_C,
			numElements, basisDim,
			&alpha,
			operationMat, numElements,
			source, 1,
			&beta,
			target, 1);

		if(cudaStreamSynchronize(blasStreamDMR[devItr]) != cudaSuccess)
		{
			fprintf(stderr, "basisProjDMR CUDA Error: Failed to synchronize BLAS stream %d.\n", devItr);
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(blasStatusDMR != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "basisProjDMR CUBLAS Error: Basis projection has failed.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		return 0;
	}
}
// Calculates the linear action of the Hessenberg matrix on the coefficients in hessSrcDMR.
// Result is stored in hessTrgDMR.
extern "C"{__host__
	int hessenbergMultDMR(void)
	{
		cuComplex alpha, beta;
		alpha.x = 1.0;
		alpha.y = 0.0;
		beta.x = 0.0;
		beta.y = 0.0;

		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "hessenbergMultDMR CUDA Error: Failed to set device to head solver.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		blasStatusDMR = cublasCgemv(blasHandleDMR[0], CUBLAS_OP_N,
			basisDimDMR + 1, basisDimDMR,
			&alpha,
			hessArrDMR, basisDimDMR + 1,
			hessSrcDMR, 1,
			&beta,
			hessWrkDMR, 1);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "hessenbergMultDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(blasStatusDMR != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "hessenbergMultDMR CUBLAS Error: Failed to apply Hessenberg operation.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		return 0;
	}
}
// Performs SV decomposition factorization of a source array, sArray.
// Results for the target singular vectors are stored in uArray, source singular vectors vArray,
// and singular values eArray. 
extern "C"{__host__
	int svDecompHessDMR(void)
	{
		int info = 0;
		int *devInfo = NULL;

		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "svDecompHessDMR CUDA Error: Failed to set device to head solver.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(cudaMalloc((void**) &devInfo, sizeof(int)) != cudaSuccess) 
		{
			fprintf(stderr, "svDecompHessDMR CUDA Error: Failed to allocate device memory for SV information.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}
		
		solverStatusDMR = cusolverDnCgesvdj(slvHandleSVDMR, vecModeSVDMR, memModeSVDMR,
			basisDimDMR + 1, basisDimDMR,
			hessArrDMR, basisDimDMR + 1,
			eHessArrDMR,
			uHessArrDMR, basisDimDMR + 1,
			vHessArrDMR, basisDimDMR,
			svWorkDMR, workSizeSVDMR,
			devInfo, geSVjPrmsDMR);

		if(cudaStreamSynchronize(slvStreamSVDMR) != cudaSuccess)
		{
			fprintf(stderr, "svDecompHessDMR CUDA Error: Failed to synchronize solver stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(solverStatusDMR != CUSOLVER_STATUS_SUCCESS)
		{
			if(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
			{
				fprintf(stderr, "svDecompHessDMR CUDA Error: Failed to copy SV information to host.\n");
				return 1;	
			}
			
			fprintf(stderr, "svDecompHessDMR CUDASOLVER Error: SV failed with error %d.\n", info);
			fullResetDMR(numDevsDMR);
			return 1;	
		}
		// Free device memory 
		cudaFree(devInfo);
		return 0;
	}
}
// Performs QR decomposition of Hessenberg matrix.
extern "C"{__host__
	int qrHessArrDMR(void)
	{
		int solverInt = 0;
		int *devInfo = NULL;

		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "qrHessArrDMR CUDA Error: Failed to set device to head solver.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(cudaMallocManaged((void**) &devInfo, sizeof(int)) != cudaSuccess) 
		{
			fprintf(stderr, "qrHessArrDMR CUDA Error: Failed to allocate device memory for device response.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(cudaMemcpyPeerAsync(cpyHessArrDMR, devListDMR[0], hessArrDMR, devListDMR[0], hessenbergSizeDMR, slvStreamQRDMR) != cudaSuccess)
		{
			fprintf(stderr, "qrHessArrDMR CUDA Error: Failed to copy Hessenberg array.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}
		// Perform QR factorization in two steps.
		// First step of QR
		solverStatusDMR = cusolverDnCgeqrf(slvHandleQRDMR, 
			basisDimDMR + 1, basisDimDMR, 
			cpyHessArrDMR, basisDimDMR + 1, 
			tCUSLVDMR, 
			wCUSLVDMR, wSizeQR, 
			devInfo);

		if(cudaStreamSynchronize(slvStreamQRDMR) != cudaSuccess)
		{
			fprintf(stderr, "qrHessArrDMR CUDA Error: Failed to synchronize solver stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(solverStatusDMR != CUSOLVER_STATUS_SUCCESS)
		{
			fprintf(stderr, "qrHessArrDMR CUSOLVER Error: Failed first step of QR factorization.\n");
			cudaMemcpy(&solverInt, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
			fprintf(stdout, "Geqrf integer: %d.\n", solverInt);
			fullResetDMR(numDevsDMR);
			return 1;	
		}
		// Export the R array. Note that R is completely filled, but only the right hand
		// (upper) part of the matrix contains useful data. This is consistent with the 
		// convention of cuda solvers for triangular matrices (toolkit v10.0.13).
		for(int colItr = 0; colItr < basisDimDMR; colItr++)
		{
			if(cudaMemcpyPeerAsync(&(rHessArrDMR[basisDimDMR * colItr]), devListDMR[0], &(cpyHessArrDMR[(basisDimDMR + 1) * colItr]), devListDMR[0], sizeof(cuComplex) * basisDimDMR, slvStreamQRDMR) != cudaSuccess)
			{
				fprintf(stderr, "qrHessArrDMR CUDA Error: Failed to copy column %d of upper triangular decomposition of Hessenberg array.\n", colItr);
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}

		if(cudaStreamSynchronize(slvStreamQRDMR) != cudaSuccess)
		{
			fprintf(stderr, "qrHessArrDMR CUDA Error: Failed to synchronize solver stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}
		// Second step of QR
		solverStatusDMR = cusolverDnCungqr(slvHandleQRDMR,
			basisDimDMR + 1, basisDimDMR, basisDimDMR,
			cpyHessArrDMR, basisDimDMR + 1,
			tCUSLVDMR,
			wCUSLVDMR, wSizeQR,
			devInfo);

		if(cudaStreamSynchronize(slvStreamQRDMR) != cudaSuccess)
		{
			fprintf(stderr, "qrHessArrDMR CUDA Error: Failed to synchronize solver stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(solverStatusDMR != CUSOLVER_STATUS_SUCCESS)
		{
			fprintf(stderr, "qrHessArrDMR CUSOLVER Error: Failed second step of QR factorization.\n");
			cudaMemcpy(&solverInt, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
			fprintf(stdout, "Ungqr integer: %d.\n", solverInt);
			fullResetDMR(numDevsDMR);
			return 1;	
		}

		if(cudaMemcpyPeerAsync(qHessArrDMR, devListDMR[0], cpyHessArrDMR, devListDMR[0], hessenbergSizeDMR, slvStreamQRDMR) != cudaSuccess)
		{
			fprintf(stderr, "qrHessArrDMR CUDA Error: Failed to copy Q basis of Hessenberg array decomposition.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(cudaStreamSynchronize(slvStreamQRDMR) != cudaSuccess)
		{
			fprintf(stderr, "qrHessArrDMR CUDA Error: Failed to synchronize solver stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}
		// Free temporary memory
		cudaFree(devInfo);   
		return 0;
	}
}
// Solves A x = s given a qr factorization of A.
extern "C"{__host__
	int qrSolveDMR(int rows, int cols, cuComplex *trgVec, cuComplex *srcVec, const cuComplex *qArray, cuComplex *rArray)
	{
		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "qrSolveDMR CUDA Error: Failed to set device to head solver.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(basisProjDMR(qArray, trgVec, srcVec, cols, rows, devListDMR[0]) != 0)
		{
			return 1;
		}

		blasStatusDMR = cublasCtrsv_v2(blasHandleDMR[0], CUBLAS_FILL_MODE_UPPER,
			CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
			cols, rArray, cols,
			srcVec, 1);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "qrSolveDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		if(blasStatusDMR != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "qrSolveDMR CUBLAS Error: Failure in QR solver.\n");
			fullResetDMR(numDevsDMR);
			return 1;	
		}

		return 0;
	}
}
// Initialize global device memory.
extern "C"{__host__
	void initDMR(int blocks, int threadsPerBlock, int *devList, int numDevs, int elements, int basisDim, int deflateDim, void(*linOpt)(void))
	{	
		if((basisDim + 1) % numDevs != 0)
		{
			fprintf(stderr, "initDMR Error: The dimension basisDim + 1 must be divisible by the number of requested devices.\n");
			return;
		}

		if(deflateDim % numDevs != 0) 
		{
			fprintf(stderr, "initDMR Error: The deflation space dimension must be divisible by the number of requested devices.\n");
			return;
		}

		if(basisDim - deflateDim < 2)
		{
			fprintf(stderr, "initDMR Error: After removing the deflation space, the basis dimension is too small to perform Arnoldi iterations.\n");
			return;
		}

		int wSizeQR1 = 0, wSizeQR2 = 0;
		linOptDMR = linOpt;
		blocksDMR = blocks;
		numElementsDMR = elements;
		threadsPerBlockDMR = threadsPerBlock;
		// Basis settings.
		numDevsDMR = numDevs;
		basisDimDMR = basisDim;
		deflateDimDMR = deflateDim;
		localBasDimDMR = (basisDim + 1) / numDevs;
		localDefDimDMR = deflateDim / numDevs;
		// Memory sizes.
		vecSizeDMR = sizeof(cuComplex) * elements;
		hessenbergSizeDMR = sizeof(cuComplex) * (basisDimDMR + 1) * basisDimDMR;
		size_t matSize = sizeof(cuComplex) * basisDimDMR * basisDimDMR;
		devListDMR = (int*)malloc(sizeof(int) * numDevs);
		// Per device memory.
		defCoeffsDMR = (cuComplex**)malloc(sizeof(cuComplex*) * numDevs);
		kryCoeffsDMR = (cuComplex**)malloc(sizeof(cuComplex*) * numDevs);
		itrBasisDMR = (cuComplex**)malloc(sizeof(cuComplex*) * numDevs);
		itrVecDMR = (cuComplex**)malloc(sizeof(cuComplex*) * numDevs);
		defBasisDMR = (cuComplex**)malloc(sizeof(cuComplex*) * numDevs);
		defWorkDMR = (cuComplex**)malloc(sizeof(cuComplex*) * numDevs);
		// Handles
		blasHandleDMR = (cublasHandle_t*)malloc(sizeof(cublasHandle_t) * numDevs);
		blasStreamDMR = (cudaStream_t*)malloc(sizeof(cudaStream_t) * numDevs); 
		// Krylov loop flag parameters
		kryLoopFlagDMR = 0;
		kryFlagCountDMR = 0;
		kryBasisLoopsDMR = 0;

		for(int devItr = 0; devItr < numDevs; devItr++)
		{	
			devListDMR[devItr] = devList[devItr];

			if(cudaSetDevice(devListDMR[devItr]) != cudaSuccess)
			{
				fprintf(stderr, "initDMR CUDA Error: Failed to set device to %d.\n", devItr);
				return;
			}

			if(devItr == 0)
			{
				if(cudaMallocManaged((void**) &hessArrDMR, hessenbergSizeDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for Hessenberg matrix on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &genWorkVecDMR, vecSizeDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate general work memory on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(numDevsDMR > 1)
				{
					if(cudaMallocManaged((void**) &sumWorkVecDMR, (numDevsDMR - 1) * vecSizeDMR) != cudaSuccess) 
					{
						fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for summing iteration vectors on head solver.\n");
						fullResetDMR(devItr);
						return;
					}
				}

				if(cudaMallocManaged((void**) &resVecDMR, vecSizeDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for residual vector on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &aSrcVecDMR, vecSizeDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for approximate solution on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &qHessArrDMR, hessenbergSizeDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for basis vectors of Hessenberg QR decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &rHessArrDMR, matSize) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for upper matrix of Hessenberg QR decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &hessSrcDMR, sizeof(cuComplex) * basisDimDMR) != cudaSuccess) 
				{
					fprintf (stderr, "initDMR CUDA Error: Failed to allocate device memory for Hessenberg source side work basis coefficients on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &hessTrgDMR, sizeof(cuComplex) * (basisDimDMR + 1)) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for Hessenberg target side basis coefficients on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &hessWrkDMR, sizeof(cuComplex) * (basisDimDMR + 1)) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for Hessenberg target side work basis coefficients on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &hessPrjDMR, sizeof(cuComplex) * (basisDimDMR + 1)) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for source basis coefficients on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &uHessArrDMR, hessenbergSizeDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for target basis of SV decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &eHessArrDMR, sizeof(float) * basisDimDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for singular values of Hessenberg on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &vHessArrDMR, matSize) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for source space of SV decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMalloc((void**) &innProdDevDMR, sizeof(cuComplex)) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for inner products on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}
				// Initialize solver handle and streams for SV decomposition.
				if(cusolverDnCreate(&slvHandleSVDMR) != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "initDMR CUSOLVER Error: Failed to create solver handle for SV decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaStreamCreateWithFlags(&slvStreamSVDMR, cudaStreamNonBlocking) != 
					cudaSuccess)
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to initialize cuSOLVER stream for SV decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cusolverDnSetStream(slvHandleSVDMR, slvStreamSVDMR) != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "initDMR CUSOLVER Error: Failed to set cuSOLVER handle to stream for SV decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;
				}
				// Configure SV
				if(cusolverDnCreateGesvdjInfo(&geSVjPrmsDMR) != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "initDMR CUSOLVER Error: Failed to set SV decomposition configuration memory location on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cusolverDnXgesvdjSetTolerance(geSVjPrmsDMR, svTolDMR) != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "initDMR CUSOLVER Error: Failed to set SV decomposition tolerance on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cusolverDnXgesvdjSetMaxSweeps(geSVjPrmsDMR, maxSVSweepsDMR) != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "initDMR CUSOLVER Error: Failed to set SV decomposition max sweeps on head solver.\n");
					fullResetDMR(devItr);
					return;	
				} 
				// Determine workspace required for SV
				if(cusolverDnCgesvdj_bufferSize(slvHandleSVDMR, vecModeSVDMR, memModeSVDMR,
					basisDimDMR + 1, basisDimDMR,
					hessArrDMR, basisDimDMR + 1,
					eHessArrDMR,
					uHessArrDMR, basisDimDMR + 1,
					vHessArrDMR, basisDimDMR,
					&workSizeSVDMR, geSVjPrmsDMR) != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "initDMR CUSOLVER Error: Failed to determine SV decomposition work size on head solver.\n");
					fullResetDMR(devItr);
					return;		
				}
				// Allocate SV workspace
				if(cudaMallocManaged((void**) &svWorkDMR, sizeof(cuComplex) * workSizeSVDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device workspace for SV decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;
				}
				// Initialize solver handle and streams for QR factorization.
				if(cusolverDnCreate(&slvHandleQRDMR) != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "initDMR CUSOLVER Error: Failed to create solver handle for QR decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaStreamCreate(&slvStreamQRDMR) != cudaSuccess)
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to initialize cuSOLVER stream for QR decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cusolverDnSetStream(slvHandleQRDMR, slvStreamQRDMR) != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "initDMR CUSOLVER Error: Failed to set cuSOLVER handle to stream for QR decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;		
				}	
				// Initialize local memory for QR solver.
				if(cudaMallocManaged((void**) &tCUSLVDMR, sizeof(cuComplex) * basisDimDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory for Tau array (QR decomposition) on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &cpyHessArrDMR, hessenbergSizeDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate device memory QR array copy on head solver.\n");
					fullResetDMR(devItr);
					return;
				}
				// Calculate required size of the work areas
				if(cusolverDnCgeqrf_bufferSize(slvHandleQRDMR,
					basisDimDMR + 1, basisDimDMR,
					cpyHessArrDMR, basisDimDMR + 1, &wSizeQR1) != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to determine work area requirements for first step of QR decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;
				}

				if(cusolverDnCungqr_bufferSize(slvHandleQRDMR,
					basisDimDMR + 1, basisDimDMR, basisDimDMR,
					cpyHessArrDMR, basisDimDMR + 1,
					tCUSLVDMR, &wSizeQR2) != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to determine work area requirements for second step of QR decomposition on head solver.\n");
					fullResetDMR(devItr);
					return;
				}
		    	// Set size of work area to maximum required size. 
				wSizeQR = (wSizeQR1 > wSizeQR2) ? wSizeQR1 : wSizeQR2;

				if(cudaMallocManaged((void**) &wCUSLVDMR, sizeof(cuComplex) * wSizeQR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate work area for QR factorization on head solver.\n");
					fullResetDMR(devItr);
					return;
				}
			}

			if(cudaStreamCreate(&(blasStreamDMR[devItr])) != cudaSuccess)
			{
				fprintf(stderr, "initDMR CUDA Error: Failed to initialize cuBLAS stream on device %d.\n", devItr);
				fullResetDMR(devItr);
				return;
			}

			if(cublasCreate(&(blasHandleDMR[devItr])) != CUBLAS_STATUS_SUCCESS) 
			{
				fprintf(stderr, "initDMR CUBLAS Error: Failed to initialize cuBLAS handle on device %d.\n", devItr);
				fullResetDMR(devItr);
				return;
			}

			if(cublasSetStream(blasHandleDMR[devItr], blasStreamDMR[devItr]) != CUBLAS_STATUS_SUCCESS) 
			{
				fprintf(stderr, "initDMR CUBLAS Error: Failed to set cuBLAS handle to cuBLAS stream on device %d.\n", devItr);
				fullResetDMR(devItr);
				return;	
			}

			if(cudaMallocManaged((void**) &(itrBasisDMR[devItr]), vecSizeDMR * localBasDimDMR) != cudaSuccess) 
			{
				fprintf(stderr, "initDMR CUDA Error: Failed to allocate memory for Krylov basis on device %d.\n", devItr);
				fullResetDMR(devItr);
				return;
			}

			if(localDefDimDMR > 0)
			{
				if(cudaMallocManaged((void**) &(defBasisDMR[devItr]), vecSizeDMR * localDefDimDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate memory for source side singular vectors on device %d.\n", devItr);
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &(defWorkDMR[devItr]), vecSizeDMR * localDefDimDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate work area for source side singular vectors on device %d.\n", devItr);
					fullResetDMR(devItr);
					return;
				}

				if(cudaMallocManaged((void**) &(defCoeffsDMR[devItr]), sizeof(cuComplex) * localDefDimDMR) != cudaSuccess) 
				{
					fprintf(stderr, "initDMR CUDA Error: Failed to allocate memory for deflation coefficients on device %d.\n", devItr);
					fullResetDMR(devItr);
					return;
				}
			}

			if(cudaMallocManaged((void**) &(kryCoeffsDMR[devItr]), sizeof(cuComplex) * localBasDimDMR) != cudaSuccess) 
			{
				fprintf(stderr, "initDMR CUDA Error: Failed to allocate memory for Krylov basis coefficients on device %d.\n", devItr);
				fullResetDMR(devItr);
				return;
			}

			if(cudaMallocManaged((void**) &(itrVecDMR[devItr]), vecSizeDMR) != cudaSuccess) 
			{
				fprintf(stderr, "initDMR CUDA Error: Failed to allocate memory for temporary solutions on device %d.\n", devItr);
				fullResetDMR(devItr);
				return;
			}
		}
		return;
	}
}

extern "C"{__host__
	void freeDMR(void)
	{	
		for(int devItr = 0; devItr < numDevsDMR; devItr++)
		{	

			if(cudaSetDevice(devListDMR[devItr]) != cudaSuccess)
			{
				fprintf(stderr, "freeDMR CUDA Error: Failed to set active device to %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return;
			}

			if(devItr == 0)
			{
				if(cudaFree(hessArrDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear memory for Hessenberg matrix on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(genWorkVecDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear general work area on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(numDevsDMR > 1)
				{
					if(cudaFree(sumWorkVecDMR) != cudaSuccess) 
					{
						fprintf(stderr, "freeDMR CUDA Error: Failed to clear sum work area on head solver.\n");
						fullResetDMR(numDevsDMR);
						return;
					}
				}

				if(cudaFree(resVecDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear memory for residual vector on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(aSrcVecDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear memory for the approximate solution on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(qHessArrDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear memory for vectors of QR Hessenberg decomposition on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(rHessArrDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear memory for upper triangular matrix of Hessenberg decomposition on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(hessSrcDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear memory for source side Hessenberg work coefficients on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(hessTrgDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear memory for Hessenberg target side coefficients on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(hessWrkDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear memory for Hessenberg target side work coefficients on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(hessPrjDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear memory for source basis coefficients on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(uHessArrDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear memory for SV target basis on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(eHessArrDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear memory for singular values on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(vHessArrDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear memory for input singular values on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(svWorkDMR) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear device SV work area on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(innProdDevDMR) != cudaSuccess)
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear device memory for inner product on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				solverStatusDMR = cusolverDnDestroyGesvdjInfo(geSVjPrmsDMR);

				if(solverStatusDMR != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "freeDMR CUBLAS Error: Failed to clear SV parameter information on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				solverStatusDMR = cusolverDnDestroy(slvHandleSVDMR);

				if(solverStatusDMR != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "freeDMR CUBLAS Error: Failed to clear SV solver handle on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaStreamDestroy(slvStreamSVDMR) != cudaSuccess)
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear SV solver stream on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(cpyHessArrDMR) != cudaSuccess)
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear QR work area on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree(tCUSLVDMR) != cudaSuccess)
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear second QR work area on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;	
				}

				if(cudaFree(wCUSLVDMR) != cudaSuccess)
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear third QR work area on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;	
				}

				solverStatusDMR = cusolverDnDestroy(slvHandleQRDMR);

				if(solverStatusDMR != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "freeDMR CUSOLVER Error: Failed to clear QR decomposition parameter information on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaStreamDestroy(slvStreamQRDMR) != cudaSuccess)
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to clear QR decomposition stream on head solver.\n");
					fullResetDMR(numDevsDMR);
					return;
				}
			}

			if(cudaFree((void*) itrBasisDMR[devItr]) != cudaSuccess) 
			{
				fprintf(stderr, "freeDMR CUDA Error: Failed to free memory for Krylov basis on device %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return;
			}

			if(localDefDimDMR > 0)
			{
				if(cudaFree((void*) defBasisDMR[devItr]) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to free memory for source side singular vectors on device %d.\n", devItr);
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree((void*) defWorkDMR[devItr]) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to free work area for singular vectors on device %d.\n", devItr);
					fullResetDMR(numDevsDMR);
					return;
				}

				if(cudaFree((void*) defCoeffsDMR[devItr]) != cudaSuccess) 
				{
					fprintf(stderr, "freeDMR CUDA Error: Failed to free memory for deflation basis coefficients on device %d.\n", devItr);
					fullResetDMR(numDevsDMR);
					return;
				}
			}

			if(cudaFree((void*) itrVecDMR[devItr]) != cudaSuccess) 
			{
				fprintf(stderr, "freeDMR CUDA Error: Failed to free memory for temporary solutions on device %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return;
			}

			if(cudaFree((void*) kryCoeffsDMR[devItr]) != cudaSuccess) 
			{
				fprintf(stderr, "freeDMR CUDA Error: Failed to free memory for Krylov basis coefficients on device %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return;
			}


			if(cublasDestroy(blasHandleDMR[devItr]) != CUBLAS_STATUS_SUCCESS) 
			{
				fprintf(stderr, "freeDMR CUBLAS Error: Failed to free cuBLAS handle on device %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return;
			}

			if(cudaStreamDestroy(blasStreamDMR[devItr]) != cudaSuccess)
			{
				fprintf(stderr, "freeDMR CUDA Error: Failed to free cuBLAS stream on device %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return;
			}
		}
		// Free CPU memory
		free(itrVecDMR);
		free(kryCoeffsDMR);
		free(defCoeffsDMR);
		free(itrBasisDMR);
		free(defBasisDMR);
		free(defWorkDMR);
		free(blasHandleDMR);
		free(blasStreamDMR);
		free(devListDMR);
		return;
	}
}
// Sums all itrVec work vectors, and stores the result on the itrVec of the primary solver device. 
// devItrUB is the device iterator upper bound. 
extern "C"{__host__
	int sumItrVecsDMR(int devItrUB)
	{

		for(int devItr = 1; devItr < devItrUB; devItr++)
		{
			if(cudaMemcpyPeerAsync(&(sumWorkVecDMR[(devItr - 1) * numElementsDMR]), devListDMR[0], itrVecDMR[devItr], devListDMR[devItr], vecSizeDMR, blasStreamDMR[devItr]) != cudaSuccess)
			{
				fprintf (stderr, "sumItrVecsDMR CUDA Error: Failed to copy coefficient information to device %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}

		for(int devItr = 0; devItr < devItrUB; devItr++)
		{
			if(cudaStreamSynchronize(blasStreamDMR[devItr]) != cudaSuccess)
			{
				fprintf(stderr, "sumItrVecsDMR CUDA Error: Failed to synchronize BLAS stream %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}

		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "sumItrVecsDMR CUDA Error: Failed to set device to head solver.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		vecAddArrDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(devItrUB - 1, numElementsDMR, itrVecDMR[0], sumWorkVecDMR);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "sumItrVecsDMR CUDA Error: Failed to synchronize BLAS stream 0.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		return 0;	
	}
}
// Loads vector components from each basis segment into the per device itrVec. 
// Vectors are generated over the entire basis, ie. both Krylov and deflation vectors
// are included. Mode -1 uses the source space basis, all other numbers use the target side basis.
extern "C"{__host__
	int genItrVecsDMR(int mode, int defDim, cuComplex *coeffs)
	{
		int localDimDef = (defDim == 0) ? 0 : localDefDimDMR;
		// Copy coefficients and sum results into iteration vectors.
		for(int devItr = 0; devItr < numDevsDMR; devItr++)
		{
			if(cudaSetDevice(devListDMR[devItr]) != cudaSuccess)
			{
				fprintf(stderr, "genItrVecsDMR CUDA Error: Failed to switch to device %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return 1;
			}

			zeroArrayDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[devItr]>>>(0, numElementsDMR, itrVecDMR[devItr]);
			// Krylov vector components.
			cudaMemcpyPeerAsync(kryCoeffsDMR[devItr], devListDMR[devItr], &(coeffs[defDim + devItr * (localBasDimDMR - localDimDef)]), devListDMR[0], sizeof(cuComplex) * (localBasDimDMR - localDimDef), blasStreamDMR[devItr]);
			// Sum basis coefficients into device itrVecDMR.
			addBasisVecsDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[devItr]>>>(localBasDimDMR - localDimDef, numElementsDMR, kryCoeffsDMR[devItr], &(itrBasisDMR[devItr][localDimDef * numElementsDMR]), itrVecDMR[devItr]);
			// Deflation vector components.
			if(localDimDef > 0)
			{
				cudaMemcpyPeerAsync(defCoeffsDMR[devItr], devListDMR[devItr], &(coeffs[devItr * localDimDef]), devListDMR[0], sizeof(cuComplex) * localDimDef, blasStreamDMR[devItr]);
				// Sum for source side vectors.
				if(mode == -1)
				{
					addBasisVecsDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[devItr]>>>(localDimDef, numElementsDMR, defCoeffsDMR[devItr], defBasisDMR[devItr], itrVecDMR[devItr]);
				}
				// Sum for target side vectors.
				else
				{
					addBasisVecsDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[devItr]>>>(localDimDef, numElementsDMR, defCoeffsDMR[devItr], itrBasisDMR[devItr], itrVecDMR[devItr]);
				}
			}
		}
		// Synchronize streams before moving to summing step
		for(int devItr = 0; devItr < numDevsDMR; devItr++)
		{
			if(cudaStreamSynchronize(blasStreamDMR[devItr]) != cudaSuccess)
			{
				fprintf(stderr, "genItrVecsDMR CUDA Error: Failed to synchronize BLAS stream %d after vector generation.\n", devItr);
				fullResetDMR(numDevsDMR);
				return 1;
			}		
		}
		// Sum all iteration vector together, storing the result in itrVecDMR[0]
		if(numDevsDMR > 1)
		{	
			if(sumItrVecsDMR(numDevsDMR) != 0)
			{
				return 1;
			}
		}

		return 0;
	}
}
// Generate source side deflation vectors. Mode zero corresponds to a first run of the Arnoldi 
// algorithm where no deflation vectors are used. 
extern "C"{__host__
	int genDeflatDMR(int mode)
	{
		int devLoc, memLoc, defDim, localDefDim;

		if(mode == 0)
		{
			localDefDim = 0;
			defDim = 0;
		}
		else
		{
			localDefDim = localDefDimDMR;
			defDim = deflateDimDMR; 
		}
		// Perform svd decompositions. 
		if(svDecompHessDMR() != 0)
		{
			return 1;
		}
		
		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "genDeflatDMR CUDA Error: Failed to set device.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		zeroArrayDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(0, basisDimDMR + 1, hessTrgDMR);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "genDeflatDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}
		// Source side
		for(int vecItr = 0; vecItr < deflateDimDMR; vecItr++)
		{
			devLoc = vecItr / localDefDimDMR;
			memLoc = vecItr % localDefDimDMR;

			if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "genDeflatDMR CUDA Error: Failed to set device.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaMemcpyPeerAsync(hessWrkDMR, devListDMR[0], &(vHessArrDMR[(vecItr + basisDimDMR - deflateDimDMR) * basisDimDMR]), devListDMR[0], sizeof(cuComplex) * basisDimDMR, blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf (stderr, "genDeflatDMR CUDA Error: Failed to copy coefficient information to work vector.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
			// Reorder coefficients
			reorderHessCoeffs<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(1, numDevsDMR, defDim, basisDimDMR + 1, localBasDimDMR - localDefDim, hessWrkDMR, hessTrgDMR);

			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "genDeflatDMR CUDA Error: Failed to reorder coefficients.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
			// Compute vector
			if(genItrVecsDMR(-1, defDim, hessTrgDMR) != 0)
			{
				return 1;
			}

			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "genDeflatDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
			// Normalize
			if(innerProdDMR(numElementsDMR, itrVecDMR[0], itrVecDMR[0], innProdDevDMR, devListDMR[0]) != 0)
			{
				return 1;
			}
			
			normalizeVecDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(-1, numElementsDMR, itrVecDMR[0], innProdDevDMR, 0.0);
			
			if(cudaMemcpyPeerAsync(&(defWorkDMR[devLoc][memLoc * numElementsDMR]), devListDMR[devLoc], itrVecDMR[0], devListDMR[0], vecSizeDMR, blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf (stderr, "genDeflatDMR CUDA Error: Failed to copy coefficient information to device %d.\n", devLoc);
				fullResetDMR(numDevsDMR);
				return 1;
			}
			
			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "genDeflatDMR CUDA Error: Failed to synchronize solver stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}
		// Update all source vectors: move vectors from work area into storage area.  
		for(int devItr = 0; devItr < numDevsDMR; devItr++)
		{
			if(cudaMemcpyPeerAsync(defBasisDMR[devItr], devListDMR[devItr], defWorkDMR[devItr], devListDMR[devItr], vecSizeDMR * localDefDimDMR, blasStreamDMR[devItr]) != cudaSuccess)
			{
				fprintf (stderr, "genDeflatDMR CUDA Error: Failed to copy singular vectors from work area to storage area on device %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}
		// Synchronize devices
		for(int devItr = 0; devItr < numDevsDMR; devItr++)
		{
			if(cudaStreamSynchronize(blasStreamDMR[devItr]) != cudaSuccess)
			{
				fprintf(stderr, "genDeflatDMR CUDA Error: Failed to synchronize BLAS stream %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}

		return 0;
	}
}
// Operates in place, deflating the vector held in itrVecDMR[0]. saveMode != 0 saves 
// projections to the Hessenberg matrix based on the Krylov iteration kryNum but does not
// renormalize the vector. 
// saveMode == 0,  orthogonalizes the vector with respect to the deflation basis, but does not
// save the projection components.
extern "C"{__host__
	int deflatorDMR(int saveMode, int kryNum) 
	{
		int hessenPos;

		for(int devItr = 1; devItr < numDevsDMR; devItr++)
		{
			if(cudaMemcpyPeerAsync(itrVecDMR[devItr], devListDMR[devItr], itrVecDMR[0], devListDMR[0], vecSizeDMR, blasStreamDMR[devItr]) != cudaSuccess)
			{
				fprintf (stderr, "deflatorDMR CUDA Error: Failed to copy inflated vector to device %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return 1;
			}	
		}

		for(int devItr = 0; devItr < numDevsDMR; devItr++)
		{
			if(basisProjDMR(itrBasisDMR[devItr], itrVecDMR[devItr], defCoeffsDMR[devItr], localDefDimDMR, numElementsDMR, devItr) != 0)
			{
				return 1;
			}
			// Position in Hessenberg array
			if(saveMode != 0)
			{
				hessenPos = (deflateDimDMR + kryNum - 1) * (basisDimDMR + 1) + devItr * localDefDimDMR;
				// Save results to Hessenberg matrix
				cudaMemcpyPeerAsync(&(hessArrDMR[hessenPos]), devListDMR[0], defCoeffsDMR[devItr], devListDMR[devItr], sizeof(cuComplex) * localDefDimDMR, blasStreamDMR[devItr]);
			}
			// Remove deflation components
			if(cudaSetDevice(devListDMR[devItr]) != cudaSuccess)
			{
				fprintf(stderr, "deflatorDMR CUDA Error: Failed to set device.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			vecScaleDiffDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[devItr]>>>(localDefDimDMR, numElementsDMR, defCoeffsDMR[devItr], itrBasisDMR[devItr], 1.0 / numDevsDMR, itrVecDMR[devItr]);
		}

		for(int devItr = 0; devItr < numDevsDMR; devItr++)
		{
			if(cudaStreamSynchronize(blasStreamDMR[devItr]) != cudaSuccess)
			{
				fprintf(stderr, "deflatorDMR CUDA Error: Failed to synchronize BLAS stream %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}

		if(numDevsDMR > 1)
		{
			if(sumItrVecsDMR(numDevsDMR) != 0)
			{
				return 1;
			}
		}

		if(saveMode == 0)
		{
			if(innerProdDMR(numElementsDMR, itrVecDMR[0], itrVecDMR[0], innProdDevDMR, devListDMR[0]) != 0)
			{
				return 1;
			}

			normalizeVecDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(-1, numElementsDMR, itrVecDMR[0], innProdDevDMR, 0.0);

			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "deflatorDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

		}

		return 0;	
	}
}
// Arnoldi algorithm, building the Krylov search space. If a defect vector is generated,
// the program zeros all remaining basis locations and exits.
extern "C"{__host__
	int arnoldiDMR(int dimDeflateSpace, cuComplex *extSrcVec, cudaStream_t extStream, int extDevNum, float defectTol)
	{	
		int prodsPerDev, offsetDevs;
		int filledLoc = 0;
		int loopBound = 0;
		int hessenPos;
		int localDefDim;
		cuComplex innerProd;
		float norm;
		int devNumS;
		int memNumS;
		int devNumT;
		int memNumT;

		localDefDim = (dimDeflateSpace > 0) ? localDefDimDMR : 0;

		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "arnoldiDMR CUDA Error: Failed to set device.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}
		// Zero Hessenberg matrix 
		zeroArrayDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(0, (basisDimDMR + 1) * basisDimDMR, hessArrDMR);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}
		// Compute deflation space. 
		for(int deflatItr = 0; deflatItr < dimDeflateSpace; deflatItr++)
		{
			devNumS = deflatItr / localDefDimDMR;
			memNumS = deflatItr % localDefDimDMR;
			devNumT = deflatItr % numDevsDMR;
			memNumT = deflatItr / numDevsDMR;

			loopBound = (memNumT > 0) ? numDevsDMR : devNumT;

			if(cudaMemcpyPeerAsync(extSrcVec, extDevNum, &(defBasisDMR[devNumS][memNumS * numElementsDMR]), devListDMR[devNumS], vecSizeDMR, extStream) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to copy source deflation vector %d, to linear operator device.\n", devNumS * localDefDimDMR + memNumS + 1);
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaSetDevice(extDevNum) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to set device to global linear operator.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			(*linOptDMR)();
			
			if(cudaMemcpyPeerAsync(itrVecDMR[0], devListDMR[0], extSrcVec, extDevNum, vecSizeDMR, extStream) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to copy global vector to head solver.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaStreamSynchronize(extStream) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize global stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaMemcpyPeerAsync(genWorkVecDMR, devListDMR[0], extSrcVec, extDevNum, vecSizeDMR, blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to external vector to general workspace.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
			
			for(int devItr = 1; devItr < loopBound; devItr++)
			{
				if(cudaMemcpyPeerAsync(itrVecDMR[devItr], devListDMR[devItr], itrVecDMR[0], devListDMR[0], vecSizeDMR, blasStreamDMR[devItr]) != cudaSuccess)
				{
					fprintf(stderr, "arnoldiDMR CUDA Error: Failed to copy iteration vector to device %d.\n", devListDMR[devItr]);
					fullResetDMR(numDevsDMR);
					return 1;
				}	
			}
			// Compute Hessenberg matrix coefficients.
			for(int devItr = 0; devItr < loopBound; devItr++)
			{
				filledLoc = (devItr < devNumT) ? (memNumT + 1) : memNumT;

				if(cudaSetDevice(devListDMR[devItr]) != cudaSuccess)
				{
					fprintf(stderr, "arnoldiDMR CUDA Error: Failed to set device to %d.\n", devItr);
					fullResetDMR(numDevsDMR);
					return 1;
				}

				zeroArrayDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[devItr]>>>(0, localDefDimDMR, defCoeffsDMR[devItr]);

				if(basisProjDMR(defWorkDMR[devItr], itrVecDMR[devItr], defCoeffsDMR[devItr], filledLoc, numElementsDMR, devItr) != 0)
				{
					return 1;
				}
			}
			// Copy deflation coefficients into Hessenberg matrix.
			if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to set device to head solver.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			zeroArrayDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(0, basisDimDMR + 1, hessWrkDMR);

			for(int devItr = 0; devItr < loopBound; devItr++)
			{
				if(cudaStreamSynchronize(blasStreamDMR[devItr]) != cudaSuccess)
				{
					fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize BLAS stream %d.\n", devItr);
					fullResetDMR(numDevsDMR);
					return 1;
				}
			}

			for(int devItr = 0; devItr < loopBound; devItr++)
			{
				filledLoc = (devItr < devNumT) ? (memNumT + 1) : memNumT;

				cudaMemcpyPeerAsync(&(hessWrkDMR[devItr * localDefDimDMR]), devListDMR[0], defCoeffsDMR[devItr], devListDMR[devItr], sizeof(cuComplex) * localDefDimDMR, blasStreamDMR[devItr]);

				if(cudaSetDevice(devListDMR[devItr]) != cudaSuccess)
				{
					fprintf(stderr, "arnoldiDMR CUDA Error: Failed to set device.\n");
					fullResetDMR(numDevsDMR);
					return 1;
				}
				// Scale iteration vectors by number of devices and remove Krylov components
				vecScaleDiffDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[devItr]>>>(filledLoc, numElementsDMR, defCoeffsDMR[devItr], defWorkDMR[devItr], 1.0 / loopBound, itrVecDMR[devItr]);
			}

			for(int devItr = 0; devItr < loopBound; devItr++)
			{
				if(cudaStreamSynchronize(blasStreamDMR[devItr]) != cudaSuccess)
				{
					fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize BLAS stream %d.\n", devItr);
					fullResetDMR(numDevsDMR);
					return 1;
				}
			}
			
			if(loopBound > 1)
			{
				if(sumItrVecsDMR(loopBound) != 0)
				{
					return 1;
				}
			}
			// Load results into Hessenberg
			hessenPos = deflatItr * (basisDimDMR + 1);

			reorderHessCoeffs<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(-1, numDevsDMR, 0, dimDeflateSpace, localDefDim, hessWrkDMR, &(hessArrDMR[hessenPos]));

			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
			// Normalize and Compute final Hessenberg component of new basis vector
			if(innerProdDMR(numElementsDMR, itrVecDMR[0], itrVecDMR[0], innProdDevDMR, devListDMR[0]) != 0)
			{
				return 1;
			}

			if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to set device.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			normalizeVecDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(-1, numElementsDMR, itrVecDMR[0], innProdDevDMR, 0.0);

			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
			// Add vector to target basis 
			if(cudaMemcpyPeerAsync(&(defWorkDMR[devNumT][memNumT * numElementsDMR]), devListDMR[devNumT], itrVecDMR[0], devListDMR[0], vecSizeDMR, blasStreamDMR[devNumT]) != cudaSuccess)
			{
				fprintf (stderr, "arnoldiDMR CUDA Error: Failed to copy new vector into basis.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}			

			if(cudaMemcpyPeerAsync(&(itrBasisDMR[devNumS][memNumS * numElementsDMR]), devListDMR[devNumS], itrVecDMR[0], devListDMR[0], vecSizeDMR, blasStreamDMR[devNumS]) != cudaSuccess)
			{
				fprintf (stderr, "arnoldiDMR CUDA Error: Failed to copy new vector into basis.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
			// Compute final Hessenberg entry
			hessenPos = (deflatItr) * (basisDimDMR + 1) + deflatItr;
			// Inner product of original image with new vector
			if(innerProdDMR(numElementsDMR, itrVecDMR[0], genWorkVecDMR, &(hessArrDMR[hessenPos]), devListDMR[0]) != 0)
			{
				return 1;
			}

			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaStreamSynchronize(blasStreamDMR[devNumS]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize BLAS stream %d.\n", devNumS);
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaStreamSynchronize(blasStreamDMR[devNumT]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize BLAS stream %d.\n", devNumT);
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}
		// Orthogonalize starting Krylov vector with respect to deflation basis.
		if(dimDeflateSpace > 0)
		{
			if(cudaMemcpyPeerAsync(itrVecDMR[0], devListDMR[0], &(itrBasisDMR[0][localDefDim * numElementsDMR]), devListDMR[0], vecSizeDMR, blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to copy starting Krylov vector.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
		
			if(deflatorDMR(0, 0) != 0)
			{
				return 1;
			}

			if(cudaMemcpyPeerAsync(&(itrBasisDMR[0][localDefDim * numElementsDMR]), devListDMR[0], itrVecDMR[0], devListDMR[0], vecSizeDMR, blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to copy starting Krylov vector.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}
		// Switch to Krylov basis computation
		if(cudaMemcpyPeerAsync(extSrcVec, extDevNum, &(itrBasisDMR[0][localDefDim * numElementsDMR]), devListDMR[0], vecSizeDMR, extStream) != cudaSuccess)
		{
			fprintf(stderr, "arnoldiDMR CUDA Error: Failed to copy initial Krylov vector.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		for(int krylovItr = 1; krylovItr < (basisDimDMR - dimDeflateSpace + 1); krylovItr++)
		{
			if(cudaSetDevice(extDevNum) != cudaSuccess)
			{
				fprintf(stderr, "deflatorDMR CUDA Error: Failed to set device to head linear operator.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			(*linOptDMR)();
			
			if(cudaMemcpyPeerAsync(itrVecDMR[0], devListDMR[0], extSrcVec, extDevNum, vecSizeDMR, extStream) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to copy global vector to solver.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaStreamSynchronize(extStream) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize global stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "deflatorDMR CUDA Error: Failed to set device to head solver.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaMemcpyPeerAsync(genWorkVecDMR, devListDMR[0], itrVecDMR[0], devListDMR[0], vecSizeDMR, blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to copy iteration vector to workspace.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to head BLAS stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}		
			// Deflate iterated vector, fills upper right hand block of Hessenberg matrix.
			if(dimDeflateSpace > 0)
			{	
				if(deflatorDMR(1, krylovItr) != 0)
				{
					return 1;
				}
			}			

			prodsPerDev = krylovItr / numDevsDMR;
			offsetDevs = krylovItr % numDevsDMR;
			loopBound = (prodsPerDev > 0) ? numDevsDMR : offsetDevs;

			for(int devItr = 1; devItr < loopBound; devItr++)
			{
				
				if(cudaMemcpyPeerAsync(itrVecDMR[devItr], devListDMR[devItr], itrVecDMR[0], devListDMR[0], vecSizeDMR, blasStreamDMR[devItr]) != cudaSuccess)
				{
					fprintf(stderr, "arnoldiDMR CUDA Error: Failed to copy iteration vector to device %d.\n", devListDMR[devItr]);
					fullResetDMR(numDevsDMR);
					return 1;
				}	
			}

			for(int devItr = 1; devItr < loopBound; devItr++)
			{
				if(cudaStreamSynchronize(blasStreamDMR[devItr]) != cudaSuccess)
				{
					fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize BLAS stream %d.\n", devItr);
					fullResetDMR(numDevsDMR);
					return 1;
				}
			}
			// Compute Hessenberg matrix coefficients with Arnoldi generated vectors.
			// Deflation vectors handled separately by deflatorDMR.
			for(int devItr = 0; devItr < loopBound; devItr++)
			{
				filledLoc = (devItr < offsetDevs) ? (prodsPerDev + 1) :	prodsPerDev;
				
				if(cudaSetDevice(devListDMR[devItr]) != cudaSuccess)
				{
					fprintf(stderr, "arnoldiDMR CUDA Error: Failed to set device to %d.\n", devItr);
					fullResetDMR(numDevsDMR);
					return 1;
				}

				zeroArrayDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[devItr]>>>(0, localBasDimDMR, kryCoeffsDMR[devItr]);

				if(basisProjDMR(&(itrBasisDMR[devItr][localDefDim * numElementsDMR]), itrVecDMR[devItr], kryCoeffsDMR[devItr], filledLoc, numElementsDMR, devItr) != 0)
				{
					return 1;
				}
			}
			// Copy Krylov coefficients into Hessenberg matrix.
			if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to set device to head solver.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			zeroArrayDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(0, basisDimDMR + 1, hessWrkDMR);

			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize BLAS stream 0.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			for(int devItr = 0; devItr < loopBound; devItr++)
			{
				filledLoc = (devItr < offsetDevs) ? (prodsPerDev + 1) :	prodsPerDev;

				if(cudaSetDevice(devListDMR[devItr]) != cudaSuccess)
				{
					fprintf(stderr, "arnoldiDMR CUDA Error: Failed to set device.\n");
					fullResetDMR(numDevsDMR);
					return 1;
				}
				
				cudaMemcpyPeerAsync(&(hessWrkDMR[devItr * (localBasDimDMR - localDefDim)]), devListDMR[0], kryCoeffsDMR[devItr], devListDMR[devItr], sizeof(cuComplex) * (localBasDimDMR - localDefDim), blasStreamDMR[devItr]);
				// Scale iteration vectors by number of devices and remove Krylov components
				vecScaleDiffDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[devItr]>>>(filledLoc, numElementsDMR, kryCoeffsDMR[devItr], &(itrBasisDMR[devItr][localDefDim * numElementsDMR]), 1.0 / loopBound, itrVecDMR[devItr]);
			}

			for(int devItr = 0; devItr < loopBound; devItr++)
			{
				if(cudaStreamSynchronize(blasStreamDMR[devItr]) != cudaSuccess)
				{
					fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize BLAS stream %d.\n", devItr);
					fullResetDMR(numDevsDMR);
					return 1;
				}
			}
			
			if(loopBound > 1)
			{
				if(sumItrVecsDMR(loopBound) != 0)
				{
					return 1;
				}
			}
			// Load results into Hessenberg
			if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to set device.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			hessenPos = (dimDeflateSpace + krylovItr - 1) * (basisDimDMR + 1) + dimDeflateSpace;

			reorderHessCoeffs<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(-1, numDevsDMR, 0, basisDimDMR - dimDeflateSpace, localBasDimDMR - localDefDim, hessWrkDMR, &(hessArrDMR[hessenPos]));

			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
			// Normalize iteration vector and compute final Hessenberg component of new basis vector
			if(innerProdDMR(numElementsDMR, itrVecDMR[0], itrVecDMR[0], innProdDevDMR, devListDMR[0]) != 0)
			{
				return 1;
			}

			if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to set device.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaMemcpy(&innerProd, innProdDevDMR, sizeof(cuComplex), cudaMemcpyDeviceToHost) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to copy inner product to host.\n");
				return 1;	
			}

			norm = sqrtf(innerProd.x);

			if(norm < defectTol)
			{
				for(int devItr = 0; devItr < numDevsDMR; devItr++)
				{
					filledLoc = (devItr < offsetDevs) ? (prodsPerDev + 1) :	prodsPerDev;
					
					if(cudaSetDevice(devListDMR[devItr]) != cudaSuccess)
					{
						fprintf(stderr, "arnoldiDMR CUDA Error: Failed to set device.\n");
						fullResetDMR(numDevsDMR);
						return 1;
					}

					zeroArrayDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[devItr]>>>((localDefDim + filledLoc) * numElementsDMR, localBasDimDMR * numElementsDMR, itrBasisDMR[devItr]);

					if(devItr == 0)
					{
						zeroArrayDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[devItr]>>>(0, numElementsDMR, itrVecDMR[devItr]);
						// Make Hessenberg full rank for QR decomposition.
						eyeArrayDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[devItr]>>>(krylovItr * (basisDimDMR + 1), basisDimDMR + 1, basisDimDMR, hessArrDMR);
					} 
				}

				for(int devItr = 0; devItr < numDevsDMR; devItr++)
				{
					if(cudaStreamSynchronize(blasStreamDMR[devItr]) != cudaSuccess)
					{
						fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize BLAS stream %d after zeroing solution basis.\n", devItr);
						fullResetDMR(numDevsDMR);
						return 1;
					}
				}

				return 0;
			}

			normalizeVecDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(-1, numElementsDMR, itrVecDMR[0], innProdDevDMR, 0.0);
			// Add vector to Krylov basis 
			if(cudaMemcpyPeerAsync(&(itrBasisDMR[offsetDevs][(localDefDim + prodsPerDev) * numElementsDMR]), devListDMR[offsetDevs], itrVecDMR[0], devListDMR[0], vecSizeDMR, blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf (stderr, "arnoldiDMR CUDA Error: Failed to copy new vector into basis.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
			// Compute final Hessenberg entry
			hessenPos = (dimDeflateSpace + krylovItr - 1) * (basisDimDMR + 1) + dimDeflateSpace + krylovItr;
			// Inner product of original image with new vector
			if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(innerProdDMR(numElementsDMR, itrVecDMR[0], genWorkVecDMR, &(hessArrDMR[hessenPos]), devListDMR[0]) != 0)
			{
				return 1;
			}

			if(cudaMemcpyPeerAsync(extSrcVec, extDevNum, itrVecDMR[0], devListDMR[0], vecSizeDMR, extStream) != cudaSuccess)
			{
				fprintf (stderr, "arnoldiDMR CUDA Error: Failed to copy to linear computer into basis.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}

			if(cudaStreamSynchronize(extStream) != cudaSuccess)
			{
				fprintf(stderr, "arnoldiDMR CUDA Error: Failed to synchronize external stream.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}

		return 0;
	}
}

extern "C"{__host__
	int updateTrgCoeffsDMR(int defDim)
	{
		int localDefDim;

		localDefDim = (defDim == 0) ? 0 : localDefDimDMR;

		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateTrgCoeffsDMR CUDA Error: Failed to set device to head solver.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		zeroArrayDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(0, basisDimDMR + 1, hessWrkDMR);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateTrgCoeffsDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		for(int devItr = 0; devItr < numDevsDMR; devItr++)
		{
			// Copy residual vector to all devices
			cudaMemcpyPeerAsync(itrVecDMR[devItr], devListDMR[devItr], resVecDMR, devListDMR[0], vecSizeDMR, blasStreamDMR[devItr]);
			// Calculate projection coefficients
			if(defDim != 0)
			{
				if(basisProjDMR(itrBasisDMR[devItr], itrVecDMR[devItr], defCoeffsDMR[devItr], localDefDim, numElementsDMR, devItr) != 0)
				{
					return 1;
				}

				cudaMemcpyPeerAsync(&(hessWrkDMR[devItr * localDefDim]), devListDMR[0], defCoeffsDMR[devItr], devListDMR[devItr], sizeof(cuComplex) * localDefDim, blasStreamDMR[devItr]);			
			}

			if(basisProjDMR(&(itrBasisDMR[devItr][localDefDim * numElementsDMR]), itrVecDMR[devItr], kryCoeffsDMR[devItr], localBasDimDMR - localDefDim, numElementsDMR, devItr) != 0)
			{
				return 1;
			}

			cudaMemcpyPeerAsync(&(hessWrkDMR[defDim + devItr * (localBasDimDMR - localDefDim)]), devListDMR[0], kryCoeffsDMR[devItr], devListDMR[devItr], sizeof(cuComplex) * (localBasDimDMR - localDefDim), blasStreamDMR[devItr]);
		}

		for(int devItr = 0; devItr < numDevsDMR; devItr++)
		{
			if(cudaStreamSynchronize(blasStreamDMR[devItr]) != cudaSuccess)
			{
				fprintf(stderr, "updateTrgCoeffsDMR CUDA Error: Failed to synchronize BLAS stream %d.\n", devItr);
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}
		// Load new projCoeffs
		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateTrgCoeffsDMR CUDA Error: Failed to set device.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		reorderHessCoeffs<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(-1, numDevsDMR, defDim, basisDimDMR + 1, localBasDimDMR - localDefDim, hessWrkDMR, hessPrjDMR);
		
		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateTrgCoeffsDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}
		
		return 0;
	}
}
// Updates the residual, approximate solution, and global vectors using the 
// results of the current solve. Outputs the current norm of the residual. 
extern "C"{__host__
	float updateAppResDMR(cuComplex *extSrcVec, cudaStream_t extStream, int extDevNum, int defDim, float solTol, float prevNorm)
	{
		cuComplex resProdHost;
		float resNorm = 0;
		int localDimDef;
		
		localDimDef = (defDim == 0) ? 0 : localDefDimDMR;
		// Calculate target coefficients. 
		if(updateTrgCoeffsDMR(defDim) != 0)
		{
			return prevNorm;
		}
		// Calculate coefficients of the residual.
		if(qrHessArrDMR() != 0)
		{
			return prevNorm;
		}

		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR CUDA Error: Failed to set device.\n");
			fullResetDMR(numDevsDMR);
			return prevNorm;
		}

		if(cudaMemcpy(hessWrkDMR, hessPrjDMR, sizeof(cuComplex) * (basisDimDMR + 1), cudaMemcpyDeviceToDevice) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR CUDA Error: Failed to copy hessPrjDMR into coefficient vector.\n");
			fullResetDMR(numDevsDMR);
			return prevNorm;
		}

		if(qrSolveDMR(basisDimDMR + 1, basisDimDMR, hessWrkDMR, hessSrcDMR, qHessArrDMR, rHessArrDMR) != 0)
		{
			return prevNorm;
		}
		// Hessenberg matrix acts on hessSrcDMR vector and places result in hessTrgDMR.
		if(hessenbergMultDMR() != 0)
		{
			return prevNorm;
		}
		// Reorder coefficients
		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR CUDA Error: Failed to set device.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		reorderHessCoeffs<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(1, numDevsDMR, defDim, basisDimDMR + 1, localBasDimDMR - localDimDef, hessWrkDMR, hessTrgDMR);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR CUDA Error: Failed to synchronize BLAS stream %d after coefficient reordering.\n", 0);
			fullResetDMR(numDevsDMR);
			return 1;
		}
		// Update residual
		if(genItrVecsDMR(1, defDim, hessTrgDMR) != 0)
		{
			return prevNorm;
		}
		// Add update to previous residual
		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR basisProjDMR CUDA Error: Failed to set device.\n");
			fullResetDMR(numDevsDMR);
			return prevNorm;
		}

		vecAddDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>> (-1, numElementsDMR, resVecDMR, itrVecDMR[0]);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR CUDA Error: Failed to synchronize main BLAS stream.\n");
			fullResetDMR(numDevsDMR);
			return prevNorm;
		}
		// Compute norm change
		if(innerProdDMR(numElementsDMR, resVecDMR, resVecDMR, innProdDevDMR, devListDMR[0]) != 0)
		{
			return prevNorm;
		}

		if(cudaMemcpy(&resProdHost, innProdDevDMR, sizeof(cuComplex), cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR CUDA Error: Failed to copy residual to host.\n");
			fullResetDMR(numDevsDMR);
			return prevNorm;
		}

		resNorm = sqrtf(resProdHost.x);
		// Set Krylov loop flag if change is smaller than solution tolerance. This switch initial 
		// vector for constructing the Krylov basis.
		if(resNorm > 0.995)
		{
			kryLoopFlagDMR = 1;
			kryFlagCountDMR++;
		}
		else
		{
			kryLoopFlagDMR = 0;
		}
		// Renormalize residual vector
		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR CUDA Error: Failed to set device.\n");
			fullResetDMR(numDevsDMR);
			return prevNorm;
		}

		normalizeVecDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(-1, numElementsDMR, resVecDMR, innProdDevDMR, 0.0);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR CUDA Error: Failed to synchronize main BLAS stream.\n");
			fullResetDMR(numDevsDMR);
			return prevNorm;
		}		
		// Update approximate solution
		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR basisProjDMR CUDA Error: Failed to set device.\n");
			fullResetDMR(numDevsDMR);
			return prevNorm;
		}

		zeroArrayDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(0, basisDimDMR + 1, hessWrkDMR);

		if(cudaMemcpyPeerAsync(hessWrkDMR, devListDMR[0], hessSrcDMR, devListDMR[0], sizeof(cuComplex) * basisDimDMR, blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf (stderr, "updateAppResDMR CUDA Error: Failed to copy coefficient information to device.\n");
			fullResetDMR(numDevsDMR);
			return prevNorm;
		}
		// Reorder coefficients
		reorderHessCoeffs<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(1, numDevsDMR, defDim, basisDimDMR + 1, localBasDimDMR - localDimDef, hessWrkDMR, hessTrgDMR);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR CUDA Error: Failed to synchronize BLAS stream %d after coefficient reordering.\n", 0);
			fullResetDMR(numDevsDMR);
			return 1;
		}
		// Update approximate solution.
		if(genItrVecsDMR(-1, defDim, hessTrgDMR) != 0)
		{
			return prevNorm;
		}

		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR CUDA Error: Failed to set device.\n");
			fullResetDMR(numDevsDMR);
			return prevNorm;
		}
		// Normalize size of update
		normalizeVecDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(2, numElementsDMR, itrVecDMR[0], innProdDevDMR, prevNorm);
		// Add update to previous approximation
		vecAddDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>> (1, numElementsDMR, aSrcVecDMR, itrVecDMR[0]);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "updateAppResDMR CUDA Error: Failed to synchronize main BLAS stream.\n");
			fullResetDMR(numDevsDMR);
			return prevNorm;
		}

		return resNorm * prevNorm;
	}
}
// Generate source coefficients and creates an initial vector for the Arnoldi method.
// mode == 0 suppose that there is no deflation space present. For all other numbers, a previously
// calculated deflation space is assumed to be present. 
extern "C" {__host__
	int arnoldiSetupDMR(int mode, int matRank, cuComplex *extSrcVec, cudaStream_t extStream, int extDevNum)
	{
		int localDefDim;
		// Fetch starting point for new basis
		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "arnoldiSetupDMR CUDA Error: Failed to set device to head inverse solver.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		localDefDim = (mode == 0) ? 0 : localDefDimDMR;
		
		if(kryLoopFlagDMR == 0 || kryFlagCountDMR % (2 * matRank / (basisDimDMR - deflateDimDMR)) == 0)
		{
			if(cudaMemcpyPeerAsync(&(itrBasisDMR[0][localDefDim * numElementsDMR]), devListDMR[0], resVecDMR, devListDMR[0], vecSizeDMR, blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf (stderr, "arnoldiSetupDMR CUDA Error: Failed to copy current residual into starting position.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}
		else
		{
			kryBasisLoopsDMR = (basisDimDMR - deflateDimDMR) / 2;
			int devLoc = (kryBasisLoopsDMR) % numDevsDMR;
			int memLoc = (kryBasisLoopsDMR) / numDevsDMR;

			if(cudaMemcpyPeerAsync(&(itrBasisDMR[0][localDefDim * numElementsDMR]), devListDMR[0], &(itrBasisDMR[devLoc][(localDefDim + memLoc) * numElementsDMR]), devListDMR[devLoc], vecSizeDMR, blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf (stderr, "arnoldiSetupDMR CUDA Error: Failed to final vector into starting position.\n");
				fullResetDMR(numDevsDMR);
				return 1;
			}
		}

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess) 
		{
			fprintf(stderr, "arnoldiSetupDMR CUDA Error: Failed to synchronize global linear operator stream.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		return 0;
	}
}
// Find an approximate solution for x for A x = s given a source s, and a linear computer A. 
// A mode value of !0 indicates that deflation spaces have been previously computed, and that 
// these vectors should be used as a starting point in the current computation. The information
// in extImgVec serves as an initial guess for the approximate solution. This vector is rewritten
// during execution. The solver returns the norm of the residual.
extern "C"{__host__
	float solverDMR(int mode, cuComplex *extSrcVec, cuComplex *extImgVec, cudaStream_t extStream, int extDevNum, float solTol, int matRank, int *numIts)
	{
		int grdItr = 0;
		int dimDeflat;
		int startDimDef;
		float resNormI, resNorm;
		cuComplex tempProdHost;
		numIts[0] = matRank * matRank;
		// Set local dimensions if previous deflation space exists.
		if(mode == 0)
		{
			startDimDef = 0;
		}
		else
		{
			startDimDef = localDefDimDMR;
		}
		// Reset Krylov flags in case of previous call without reinitialization.
		kryLoopFlagDMR = 0;
		kryBasisLoopsDMR = 0;

		if(cudaSetDevice(extDevNum) != cudaSuccess)
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to switch to linear operator device.\n");
			fullResetDMR(numDevsDMR);
			return 1.0;
		}

		if(cudaDeviceSynchronize() != cudaSuccess)
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to synchronize linear operator device.\n");
			fullResetDMR(numDevsDMR);
			return 1.0;
		}
		
		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to set device to head inverse solver.\n");
			fullResetDMR(numDevsDMR);
			return 1.0;
		}

		if(cudaMemcpyPeerAsync(resVecDMR, devListDMR[0], extImgVec, extDevNum, vecSizeDMR, blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf (stderr, "solverDMR CUDA Error: Failed to copy image into residual.\n");
			fullResetDMR(numDevsDMR);
			return 1.0;
		}

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to synchronize BLAS stream 0.\n");
			fullResetDMR(numDevsDMR);
			return 1.0;
		}
		
		if(innerProdDMR(numElementsDMR, resVecDMR, resVecDMR, innProdDevDMR, devListDMR[0]) != 0)
		{
			fprintf(stderr, "solverDMR Error: innerProdDMR exit.\n");
			return 1.0;
		}
		
		if(cudaMemcpy(&tempProdHost, innProdDevDMR, sizeof(cuComplex), cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to copy residual to host.\n");
			return 1.0;	
		}
		
		resNormI = sqrtf(tempProdHost.x);
		// Set initial approximation to guess.
		if(cudaMemcpyPeerAsync(aSrcVecDMR, devListDMR[0], extSrcVec, extDevNum, vecSizeDMR, blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to copy initial guess into approximation.\n");
			fullResetDMR(numDevsDMR);
			return resNormI;
		}

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess) 
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
			fullResetDMR(numDevsDMR);
			return resNormI;
		}
		// Compute initial residual, and save the result as the starting point for the 
		// first application of the Arnoldi algorithm.
		// Apply linear operator to the global vector.
		if(cudaSetDevice(extDevNum) != cudaSuccess)
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to set device to linear operator.\n");
			fullResetDMR(numDevsDMR);
			return 1;
		}

		(*linOptDMR)();

		if(cudaStreamSynchronize(extStream) != cudaSuccess)
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to synchronize external stream.\n");
			fullResetDMR(numDevsDMR);
			return 1.0;
		}

		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to set device to head inverse solver.\n");
			fullResetDMR(numDevsDMR);
			return resNormI;
		}

		if(cudaMemcpyPeerAsync(itrVecDMR[0], devListDMR[0], extSrcVec, extDevNum, vecSizeDMR, blasStreamDMR[0]) != cudaSuccess)
		{
			fprintf (stderr, "solverDMR CUDA Error: Failed to copy initial guess into approximation.\n");
			fullResetDMR(numDevsDMR);
			return resNormI;
		}

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess) 
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
			fullResetDMR(numDevsDMR);
			return resNormI;
		}
		// Find initial residual.
		vecAddDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(-1, numElementsDMR, resVecDMR, itrVecDMR[0]);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess) 
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to synchronize head BLAS stream.\n");
			fullResetDMR(numDevsDMR);
			return resNormI;
		}
		// Normalize the residual and save the result.
		if(innerProdDMR(numElementsDMR, resVecDMR, resVecDMR, innProdDevDMR, devListDMR[0]) != 0)
		{
			fprintf(stderr, "solverDMR Error: innerProdDMR exit.\n");
			return resNormI;
		}

		if(cudaMemcpy(&tempProdHost, innProdDevDMR, sizeof(cuComplex), cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to copy residual to host.\n");
			return resNormI;	
		}

		resNorm = sqrtf(tempProdHost.x);
	
		if(resNorm < resNormI * solTol)
		{
			if(cudaMemcpyPeerAsync(extSrcVec, extDevNum, aSrcVecDMR, devListDMR[0], vecSizeDMR, blasStreamDMR[0]) != cudaSuccess)
			{
				fprintf(stderr, "solverDMR CUDA Error: Failed to copy initial guess into approximation.\n");
				fullResetDMR(numDevsDMR);
				return resNormI;
			}

			fprintf(stdout, "Initial guess within solution tolerance.\n");
			return resNorm;
		}
		// Normalize residual as first vector for Arnoldi algorithm. 
		if(cudaSetDevice(devListDMR[0]) != cudaSuccess)
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to set device to head inverse solver.\n");
			fullResetDMR(numDevsDMR);
			return resNormI;
		}

		normalizeVecDMR<<<blocksDMR, threadsPerBlockDMR, 0, blasStreamDMR[0]>>>(-1, numElementsDMR, resVecDMR, innProdDevDMR, 0.0);

		if(cudaStreamSynchronize(blasStreamDMR[0]) != cudaSuccess) 
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to synchronize head BLAS stream in source construction.\n");
			fullResetDMR(numDevsDMR);
			return resNormI;
		}
		// Iterate Arnoldi method
		while(resNorm > solTol * resNormI)
		{
			if(grdItr > matRank)
			{
				fprintf (stderr, "solverDMR Inverse Solve Error: GRD has failed to converge to a solution after %d iterations.\n", grdItr);
				numIts[0] = grdItr;
				return resNorm / resNormI;
			}

			if(resNorm / resNormI > 1.0)
			{
				fprintf (stderr, "solverDMR Inverse Solve Error: GRD Runaway!\n");
				fprintf (stderr, "\n");				
				numIts[0] = grdItr;
				return resNorm / resNormI;
			}

			if(grdItr == 0 && startDimDef == 0)
			{
				dimDeflat = 0;	
			}
			else
			{
				dimDeflat = deflateDimDMR;
			}
			
			if(arnoldiSetupDMR(dimDeflat, matRank, extSrcVec, extStream, extDevNum) != 0)
			{
				fprintf(stderr, "solverDMR Error: arnoldiSetupDMR exit.\n");
				return resNormI;
			}
			// Construct Krylov Space.
			if(arnoldiDMR(dimDeflat, extSrcVec, extStream, extDevNum, solTol) != 0)
			{
				return resNormI;
			}		 
				
			resNorm = updateAppResDMR(extSrcVec, extStream, extDevNum, dimDeflat, solTol, resNorm);

			if(localDefDimDMR > 0)
			{
				if(grdItr == 0 && startDimDef == 0)
				{
					if(genDeflatDMR(0) != 0)
					{
						fprintf(stderr, "solverDMR Error: genDeflatDMR exit.\n");
						return resNormI;
					}
				}
				else
				{
					if(genDeflatDMR(1) != 0)
					{
						fprintf(stderr, "solverDMR Error: genDeflatDMR exit.\n");
						return resNormI;
					}
				}
			}

			grdItr++;
		}
		// Export number of iterations.
		numIts[0] = grdItr - 1;

		if(cudaSetDevice(extDevNum) != cudaSuccess)
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to set device to global linear operator.\n");
			fullResetDMR(numDevsDMR);
			return resNormI;
		}

		if(cudaMemcpyPeerAsync(extSrcVec, extDevNum, aSrcVecDMR, devListDMR[0], vecSizeDMR, extStream) != cudaSuccess)
		{
			fprintf (stderr, "solverDMR CUDA Error: Failed to export solution approximation.\n");
			fullResetDMR(numDevsDMR);
			return resNormI;
		}

		if(cudaStreamSynchronize(extStream) != cudaSuccess) 
		{
			fprintf(stderr, "solverDMR CUDA Error: Failed to synchronize global linear operator stream.\n");
			fullResetDMR(numDevsDMR);
			return resNormI;
		}

		return resNorm / resNormI;
	}
}
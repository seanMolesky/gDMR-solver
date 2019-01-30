#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include "gDMR.h"
// GPU memory locations
cuComplex *srcLS;
cuComplex *imgLS;
cuComplex *optLS;
cudaStream_t laStreamLS;
cublasHandle_t laHandleLS;
cudaError_t cuErrorLS;
const char *errorStringLS;
size_t matSizeLS;
size_t vecSizeLS;
int numEleLS;
int devNumLS;
// Internally accessible functions, device compatible.
__global__
void arrayInitLS(int rows, int cols, cuComplex *array)
{
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;
	int matNum = rows * cols;
	int rowNum = 0;
	int colNum = 0;
	
	for (int i = localId; i < matNum; i += stride)
	{
		rowNum = i / rows;
		colNum = i % rows;

		if (rowNum == colNum)
		{
			array[i].x = 1.0;
			array[i].y = 0.0;
		}
		else if (colNum == rowNum + 1)
		{
			array[i].x = 0.0;
			array[i].y = 0.0;	
		}
		else
		{
			array[i].x = 0.0;
			array[i].y = 0.0;
		}
	}

	return;	
}


// Externally accessible functions.
extern "C"{__host__
	void resetLS(void)
	{
		if (cudaSetDevice(0) != cudaSuccess)
		{
			fprintf(stderr, "resetLS CUDA Error: Failed to reset LS device.\n");
			return;
		}			

		if (cudaDeviceReset() != cudaSuccess)
		{
			fprintf(stderr, "fullResetDMR CUDA Error: Failed to reset device.\n");
			return;
		}
		
		return;
	}
}

extern "C"{__host__
	void printCuErrorLS(const char *cuErrorString)
	{
		fprintf(stderr, "%s", cuErrorString);
		fprintf(stderr, ".\n");
		resetLS();
		return;
	}
}

extern "C"{__host__
	void testFuncLS(void)
	{	
		cuComplex prod;
		prod.x = 0.1;
		prod.y = 0.0;
		fprintf(stdout, "Test output not working %.2f.\n", sqrtf(prod.x));
		return;
	}
}

extern "C" {__host__
	void linOptLS(void)
	{	
		if (cudaSetDevice(devNumLS) != cudaSuccess)
		{
			fprintf(stderr, "optLS CUDA Error: Failed to set device.\n");
			resetLS();
			return;
		}

		cuComplex alpha, beta;
		alpha.x = 1.0;
		beta.x = 0.0;
		alpha.y = 0.0;
		beta.y = 0.0;

		if(cublasCgemv(laHandleLS, CUBLAS_OP_N, 
			numEleLS, numEleLS,
			&alpha,
			optLS, numEleLS,
			srcLS, 1,
			&beta,
			srcLS, 1) != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "optLS CUBLAS Error: Failed to preform matrix multiplication.\n");
			resetLS();
			return;	
		}

		return;
	}
}

extern "C"{__host__ 
	void initLS(int blocksDMR, int threadsPerBlockDMR, int devNumLO, int cells, int *devListDMR, int numDevsDMR, int basisSizeDMR, int deflatSizeDMR)
	{	
		numEleLS = cells;
		devNumLS = devNumLO;
		matSizeLS = sizeof(cuComplex) * numEleLS * numEleLS;
		vecSizeLS = sizeof(cuComplex) * numEleLS;

		if (cudaSetDevice(devNumLO) != cudaSuccess)
		{
			fprintf(stderr, "initLS CUDA Error: Failed to set device.\n");
			resetLS();
			return;
		}

		if (cudaMallocManaged((void**) &optLS, matSizeLS) != cudaSuccess) 
		{
			fprintf (stderr, "initLS CUDA Error: Failed to allocate device memory for global matrix.\n");
			resetLS();
			return;
		}

		if (cudaMallocManaged((void**) &srcLS, vecSizeLS) != cudaSuccess) 
		{
			fprintf(stderr, "initLS CUDA Error: Failed to allocate device memory for srcLS.\n");
			resetLS();
			return;
		}

		if (cudaMallocManaged((void**) &imgLS, vecSizeLS) != cudaSuccess) 
		{
			fprintf(stderr, "initLS CUDA Error: Failed to allocate device memory for imgLS.\n");
			resetLS();
			return;
		}

		if(cudaStreamCreate(&laStreamLS) != cudaSuccess)
		{
			fprintf(stderr, "initLS CUDA Error: Failed to initialize cublas stream.\n");
			resetLS();
			return;
		}

		if (cublasCreate(&laHandleLS) != CUBLAS_STATUS_SUCCESS) 
		{
			fprintf(stderr, "initLS CUBLAS Error: Failed to initialize cublas handle.\n");
			resetLS();
			return;
		}

		if(cublasSetStream(laHandleLS, laStreamLS) != CUBLAS_STATUS_SUCCESS) 
		{
			fprintf(stderr, "initLS CUBLAS Error: Failed to associate cublas handle and stream.\n");
			resetLS();
			return;	
		}

		arrayInitLS<<<blocksDMR, threadsPerBlockDMR, 0, laStreamLS>>>(numEleLS, numEleLS, optLS);

		initDMR(blocksDMR, threadsPerBlockDMR, devListDMR, numDevsDMR, numEleLS, basisSizeDMR, deflatSizeDMR, linOptLS);

		return;
	}
}

extern "C"{__host__
	void freeLS(void)
	{	
		if (cudaSetDevice(devNumLS) != cudaSuccess)
		{
			fprintf(stderr, "freeLS CUDA Error: Failed to set device.\n");
			resetLS();
			return;
		}

		if (cudaFree(srcLS) != cudaSuccess) 
		{
			fprintf(stderr, "freeLS CUDA Error: Failed to free device memory for srcLS.\n");
			resetLS();
			return;
		}

		if (cudaFree(optLS) != cudaSuccess) 
		{
			fprintf(stderr, "freeLS CUDA Error: Failed to free device memory for global matrix.\n");
			resetLS();
			return;
		}

		if (cudaFree(imgLS) != cudaSuccess) 
		{
			fprintf (stderr, "freeLS CUDA Error: Failed to free device memory for imgLS.\n");
			resetLS();
			return;
		}

		if (cublasDestroy(laHandleLS) != CUBLAS_STATUS_SUCCESS) 
		{
			fprintf (stderr, "freeLS CUBLAS Error: Failed to destroy cublas handle.\n");
			resetLS();
			return;
		}

		if(cudaStreamDestroy(laStreamLS) != cudaSuccess)
		{
			fprintf(stderr, "freeLS CUDA Error: Failed to destroy cublas stream.\n");
			resetLS();
			return;
		} 

		freeDMR();

		return;
	}
}

extern "C" {__host__
	void impImgLS(float _Complex *cImg)
	{
		if (cudaSetDevice(devNumLS) != cudaSuccess)
		{
			fprintf(stderr, "impImgLS CUDA Error: Failed to set device.\n");
			resetLS();
			return;
		}

		if(cudaMemcpy(imgLS, cImg, vecSizeLS, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			fprintf(stderr, "impImgLS CUDA Error: Failed to copy memory from host to device.\n");
			resetLS();
			return;
		}

		return;
	}
}

extern "C" {__host__
	void impSrcLS(float _Complex *cSrc)
	{	
		if (cudaSetDevice(devNumLS) != cudaSuccess)
		{
			fprintf(stderr, "impSrcLS CUDA Error: Failed to set device.\n");
			resetLS();
			return;
		}

		if(cudaMemcpy(srcLS, cSrc, vecSizeLS, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			fprintf(stderr, "impSrcLS CUDA Error: Failed to copy memory from host to device.\n");
			resetLS();
			return;
		}

		return;
	}
}

extern "C" {__host__
	void impOptLS(float _Complex *cOpt)
	{
		if (cudaSetDevice(devNumLS) != cudaSuccess)
		{
			fprintf(stderr, "impOptLS CUDA Error: Failed to set device.\n");
			resetLS();
			return;
		}

		if(cudaMemcpy(optLS, cOpt, matSizeLS, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			fprintf(stderr, "impOptLS CUDA Error: Failed to copy memory from host to device.\n");
			resetLS();
			return;
		}

		return;
	}
}

extern "C" {__host__
	void expImgLS(float _Complex *cImg)
	{	
		if (cudaSetDevice(devNumLS) != cudaSuccess)
		{
			fprintf(stderr, "expImgLS CUDA Error: Failed to set device.\n");
			resetLS();
			return;
		}

		cuErrorLS = cudaMemcpy(cImg, imgLS, vecSizeLS, cudaMemcpyDeviceToHost); 

		if(cuErrorLS != cudaSuccess)
		{
			fprintf(stderr, "expImgLS CUDA Error: Failed to copy memory from device to host.\n");
			errorStringLS = cudaGetErrorString(cuErrorLS);
			printCuErrorLS(errorStringLS);
			resetLS();
			return;
		}

		return;
	}
}

extern "C" {__host__
	void expSrcLS(float _Complex *cSrc)
	{	
		if (cudaSetDevice(devNumLS) != cudaSuccess)
		{
			fprintf(stderr, "expSrcLS CUDA Error: Failed to set device.\n");
			resetLS();
			return;
		}

		cuErrorLS = cudaMemcpy(cSrc, srcLS, vecSizeLS, cudaMemcpyDeviceToHost); 

		if(cuErrorLS != cudaSuccess)
		{
			fprintf(stderr, " expSrcLS CUDA Error: Failed to copy memory from device to host.\n");
			errorStringLS = cudaGetErrorString(cuErrorLS);
			printCuErrorLS(errorStringLS);
			resetLS();
			return;
		}

		return;
	}
}

extern "C" {__host__
	void expOptLS(float _Complex *cOpt)
	{	
		if (cudaSetDevice(devNumLS) != cudaSuccess)
		{
			fprintf(stderr, "expOptLS CUDA Error: Failed to set device.\n");
			resetLS();
			return;
		}

		cuErrorLS = cudaMemcpy(cOpt, optLS, matSizeLS, cudaMemcpyDeviceToHost); 

		if(cuErrorLS != cudaSuccess)
		{
			fprintf(stderr, "expOptLS CUDA Error: Failed to copy memory from device to host.\n");
			errorStringLS = cudaGetErrorString(cuErrorLS);
			printCuErrorLS(errorStringLS);
			resetLS();
			return;
		}

		return;
	}
}
// Solve linear system using gDMR, returning the residual norm of the solution.
// A mode value of !0 indicates that the solver has been previously called on a similar linear 
// system, and should use the existing deflation space in its first iteration.
extern "C" {__host__
	float solveLS(int mode, float solTol, int matRank, int *numIts)
	{
		return solverDMR(mode, srcLS, imgLS, laStreamLS, devNumLS, solTol, matRank, numIts);
	}
}
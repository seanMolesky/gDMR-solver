#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include "gDMRC.h"

float maxRandProg = 100.0 / 1024.0;

float randFloat(void)
{
	return (float)rand() / (float)(RAND_MAX / maxRandProg) - (float)maxRandProg / 2;
}

void arrayInitRand(int rows, int cols, float _Complex *array)
{
	
	for(int i = 0; i < rows * cols; i++)
	{
		array[i] = randFloat() + _Complex_I * randFloat();
	}

	return;	
}

void unitNorm(int lenVec, float _Complex *vec)
{	
	float innerProd = 0;
	float invNorm;

	for(int i = 0; i < lenVec; i++)
	{
		innerProd = creal(vec[i]) * creal(vec[i]) + cimag(vec[i]) * cimag(vec[i]);
	}

	invNorm = 1.0 / sqrtf(innerProd);

	for(int i = 0; i < lenVec; i++)
	{
		vec[i] = creal(vec[i]) * invNorm + _Complex_I * cimag(vec[i]);
	}

	return;
}

void arrayInitZero(int rows, int cols, float _Complex *array)
{
	
	for(int i = 0; i < rows * cols; i++)
	{
		array[i] = 0.0;
	}

	return;	
}

void arrayInitOne(int rows, int cols, float _Complex *array)
{
	
	for(int i = 0; i < rows * cols; i++)
	{
		if(i == 0)
		{
			array[i] = 1.0;
		}
		else
		{
			array[i] = 0.0;
		}
	}

	return;	
}

void arrayInitLaplacian(int rows, int cols, float _Complex *array)
{
	int rowNum;
	int colNum;

	for(int i = 0; i < rows * cols; i++)
	{
		rowNum = i / rows;
		colNum = i % rows;

		if(rowNum == colNum)
		{
			array[i] = - 5.0 / 2.0;
		}
		else if ((rowNum == colNum + 1) || (rowNum == colNum - 1))
		{
			array[i] = 4.0 / 3.0;
		}
		else if ((rowNum == colNum + 1) || (rowNum == colNum - 1))
		{
			array[i] = - 1.0 / 12.0;
		}
		else
		{
			array[i] = 0.0;
		}
	}

	return;	
}

void printArray(int elements, float _Complex *array)
{
	for(int i = 0; i < elements; i++)
	{
		fprintf(stdout, "%5.4f+i%5.4f\n", creal(array[i]), cimag(array[i]));
	}

	fprintf(stdout, "\n");
	return;
}

void initNewRHS(int elements, float _Complex *vecSrc, float _Complex *vecImg)
{
	arrayInitRand(elements, 1, vecSrc);
	// Move random source to GPU.
	impSrcLS(vecSrc);
	// Perform linear operation, generating an acceptable image vector on the GPU.
	linOptLS();
	// Bring image (located on source) back to CPU for normalization.
	expSrcLS(vecSrc);
	// Normalize image size and export to GPU.
	unitNorm(elements, vecSrc);
	impImgLS(vecSrc);
	// Set initial guess to zero.
	arrayInitZero(elements, 1, vecSrc);
	impSrcLS(vecSrc);
	return;
}

int main(void)
{	
	float _Complex *cVecImg, *cMatLO, *cVecSrc;
	int elements = 1024;
	float residual;
	// Solution tolerance must be large to find solutions for uniform random matrix.
	float solutionTolerance = 0.00001;
	// Number of tests to perform.
	int numTest = 5;
	// Container for the number of iterations used in a given call of the inverse solver.
	int numIts;
	// GPU settings
	int devNumLO = 0;
	int blocksDMR = 32, threadsPerBlockDMR = 512, basisSizeDMR = 127, deflatSizeDMR = 0;
	// Device list for inverse solver. Virtual GPUs can be added by repeating the same number in 
	// the device list. For example, if devListDMR[] = {0,0,0}, numDevsDMR = 3, the solver will 
	// run as if three devices are present. (This does not lead to faster computation, but may
	// be useful for testing).
	int devListDMR[] = {0}, numDevsDMR = 1;
	// Create random seed.
	srand((unsigned int)time(NULL));
	// Allocate memory
	cMatLO = (float _Complex*)malloc(sizeof(float _Complex) * elements * elements);
	cVecImg = (float _Complex*)malloc(sizeof(float _Complex) * elements);
	cVecSrc = (float _Complex*)malloc(sizeof(float _Complex) * elements);
	// Initialize linear system
	initLS(blocksDMR, threadsPerBlockDMR, devNumLO, elements, (int*) devListDMR, numDevsDMR, basisSizeDMR, deflatSizeDMR);
	// Initialize linear operator values.
	arrayInitLaplacian(elements, elements, cMatLO);
	impOptLS(cMatLO);

	for(int i = 0; i < numTest; i++)
	{
		initNewRHS(elements, cVecSrc, cVecImg);
		residual = solveLS(i, solutionTolerance, elements, &numIts);
		fprintf(stdout, "gDMR Residual: %7.6f after %d iterations.\n", residual, numIts);
		expSrcLS(cVecSrc);
		linOptLS();
		expSrcLS(cVecSrc);
	}
	// Free memory on GPU.
	freeLS();
	// Free allocated memory
	free(cVecImg);
	free(cMatLO);
	free(cVecSrc);
	return 0;	
}
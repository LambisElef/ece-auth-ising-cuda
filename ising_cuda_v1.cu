/*
 * ising_cuda_v1.cu
 *
 *  Created on: Dec 26, 2019
 *      Author: Lambis
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512
#define threadsNum 64

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Kernel Function.
__global__ void spin(int *G, double *w, int *newG, int n) {

    // Calculates Atomic Spin index.
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    // Checks for out of bounds indexing and if so quits.
    if (index >= n*n)
        return;

    double weightSum = 0;

    // Calculates weight contribution for each neighboring Atomic Spin and sums it.
    weightSum += w[0] * G[((index/n - 2 + n)%n) * n + (index - 2 + n)%n];
    weightSum += w[1] * G[((index/n - 2 + n)%n) * n + (index - 1 + n)%n];
    weightSum += w[2] * G[((index/n - 2 + n)%n) * n + (index)%n];
    weightSum += w[3] * G[((index/n - 2 + n)%n) * n + (index + 1 + n)%n];
    weightSum += w[4] * G[((index/n - 2 + n)%n) * n + (index + 2 + n)%n];

    weightSum += w[5] * G[((index/n - 1 + n)%n) * n + (index - 2 + n)%n];
    weightSum += w[6] * G[((index/n - 1 + n)%n) * n + (index - 1 + n)%n];
    weightSum += w[7] * G[((index/n - 1 + n)%n) * n + (index)%n];
    weightSum += w[8] * G[((index/n - 1 + n)%n) * n + (index + 1 + n)%n];
    weightSum += w[9] * G[((index/n - 1 + n)%n) * n + (index + 2 + n)%n];

    weightSum += w[10] * G[((index/n + n)%n) * n + (index - 2 + n)%n];
    weightSum += w[11] * G[((index/n + n)%n) * n + (index - 1 + n)%n];
    // w[12] is not contributing anything. It's the current Atomic Spin.
    weightSum += w[13] * G[((index/n + n)%n) * n + (index + 1 + n)%n];
    weightSum += w[14] * G[((index/n + n)%n) * n + (index + 2 + n)%n];

    weightSum += w[15] * G[((index/n + 1 + n)%n) * n + (index - 2 + n)%n];
    weightSum += w[16] * G[((index/n + 1 + n)%n) * n + (index - 1 + n)%n];
    weightSum += w[17] * G[((index/n + 1 + n)%n) * n + (index)%n];
    weightSum += w[18] * G[((index/n + 1 + n)%n) * n + (index + 1 + n)%n];
    weightSum += w[19] * G[((index/n + 1 + n)%n) * n + (index + 2 + n)%n];

    weightSum += w[20] * G[((index/n + 2 + n)%n) * n + (index - 2 + n)%n];
    weightSum += w[21] * G[((index/n + 2 + n)%n) * n + (index - 1 + n)%n];
    weightSum += w[22] * G[((index/n + 2 + n)%n) * n + (index)%n];
    weightSum += w[23] * G[((index/n + 2 + n)%n) * n + (index + 1 + n)%n];
    weightSum += w[24] * G[((index/n + 2 + n)%n) * n + (index + 2 + n)%n];

    //! Can it be done more efficiently?
    if (weightSum > 0.0001)
        newG[index] = 1;
    else if (weightSum < -0.0001)
        newG[index] = -1;
    else
        newG[index] = G[index];

}

// Kernel Function that checks whether the new Atomic Spins Matrix is the same as the old one.
__global__ void check(int *G, int *newG, int n, int *same) {

    // Calculates Atomic Spin index.
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    // Checks for out of bounds indexing and if so quits.
    if (index >= n*n)
        return;

    if (G[index] != newG[index])
	*same = 0;

}

void ising(int *G, double *w, int k, int n) {

    // Creates and transfers the Weight Matrix to GPU memory.
    double *w_d;
    int w_size = 25*sizeof(double);
    gpuErrchk( cudaMalloc((void **) &w_d, w_size) );
    gpuErrchk( cudaMemcpy(w_d, w, w_size, cudaMemcpyHostToDevice) );

    // Creates and transfers the Atomic Spins Matrix to GPU memory.
    int *G_d;
    int G_size = n*n*sizeof(int);
    gpuErrchk( cudaMalloc((void **) &G_d, G_size) );
    gpuErrchk( cudaMemcpy(G_d, G, G_size, cudaMemcpyHostToDevice) );

    // Creates the new Atomic Spins Matrix to GPU memory.
    int *newG_d;
    gpuErrchk( cudaMalloc((void **) &newG_d, G_size) );

    // Creates and transfers a flag that states whether the new Atomic Spins Matrix and the old are the same to GPU memory.
    int same = 1;
    int *same_d;
    gpuErrchk( cudaMalloc((void **) &same_d, sizeof(int)) );
    gpuErrchk( cudaMemcpy(same_d, &same, sizeof(int), cudaMemcpyHostToDevice) );

    // Creates a temporary variable for Atomic Spins Matrices' pointers swapping.
    int *temp_d;

    // Checks if function has to be iterated.
    while (k != 0) {
        // Calls the kernel function balancing load to (n*n+threadsNum-1)/threadsNum blocks with threadsNum threads each.
        spin<<<(n*n+threadsNum-1)/threadsNum,threadsNum>>>(G_d, w_d, newG_d, n);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        k--;
        
	check<<<(n*n+threadsNum-1)/threadsNum,threadsNum>>>(G_d, newG_d, n, same_d);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaMemcpy(&same, same_d, sizeof(int), cudaMemcpyDeviceToHost) );
	if (same)
	    break;

        // Atomix Spin Matrices' pointers swapping.
        temp_d = G_d;
        G_d = newG_d;
        newG_d = temp_d;
    }

    // Copies data from GPU to CPU memory.
    gpuErrchk( cudaMemcpy(G, G_d, G_size, cudaMemcpyDeviceToHost) );

    // Cleanup.
    gpuErrchk( cudaFree(w_d) );
    gpuErrchk( cudaFree(G_d) );
    gpuErrchk( cudaFree(newG_d) );

}

int main() {

    // Weight Matrix.
    double w[] = { 0.004, 0.016, 0.026, 0.016, 0.004,
                   0.016, 0.071, 0.117, 0.071, 0.016,
                   0.026, 0.117, 0.000, 0.117, 0.026,
                   0.016, 0.071, 0.117, 0.071, 0.016,
                   0.004, 0.016, 0.026, 0.016, 0.004 };

    // Number of dimensions for the square Atomic Spins Matrix.
    int n = N;

    // Allocates memory for the Atomic Spins Matrix.
    int *G = (int *)malloc(n*n * sizeof(int));


    // Randomizes seed.
    srand(time(NULL));

    // Fills the Atomic Spins Matrix with "-1" and "1" values from a uniform distribution.
    for (int i=0; i<n*n; i++)
        G[i] = ((rand() % 2) * 2) - 1;

    /*
    // Reads configuration file.
    size_t readStatus;
    FILE *conf_init = fopen("conf-init.bin","rb");
    int initG[n*n];
    readStatus = fread(&initG, sizeof(int), n*n, conf_init);
    if (readStatus != n*n)
        printf("Could not read conf-init.bin file.\n");
    fclose(conf_init);

    // Fills the Atomic Spins Matrix with "-1" and "1" values from configuration file.
    for (int i=0; i<n*n; i++)
        G[i] = initG[i];
    */

    ising(G, w, 10, n);

    /*
    // Reads configuration file for state after one iteration.
    size_t readStatus1;
    FILE *conf_1 = fopen("conf-1.bin","rb");
    int G1[n*n];
    readStatus1 = fread(&G1, sizeof(int), n*n, conf_1);
    if (readStatus1 != n*n)
        printf("Could not read conf-1.bin file.\n");
    fclose(conf_1);

    // Checks for errors.
    int errorsNum = 0;
    for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
            if (G[i*n+j] != G1[i*n+j])
                errorsNum++;
    if (errorsNum == 0)
        printf("Correct Results!\n");
    else
        printf("Wrong Results. Number of errors: %d\n", errorsNum);


    // Checks the results.
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (G[i*n+j] == G1[i*n+j])
                printf("=");
            else
                printf("!");
        }
        printf("\n");
    }
    printf("\n\n");
    */

    return 0;

}

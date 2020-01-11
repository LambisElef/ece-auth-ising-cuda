/*
 * ising_sequential.c
 *
 *  Created on: Dec 14, 2019
 *      Author: Lambis
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512

// This function will be turned into a cuda kernel.
void spin(int *G, double *w, int *newG, int n) {

    // To calculate every new Atomic Spin value do:
    for (int i=0; i<n*n; i++) {
        double weightSum = 0;

        // Calculates weight contribution for each neighboring Atomic Spin and sums it.
        weightSum += w[0] * G[((i/n - 2 + n)%n) * n + (i - 2 + n)%n];
        weightSum += w[1] * G[((i/n - 2 + n)%n) * n + (i - 1 + n)%n];
        weightSum += w[2] * G[((i/n - 2 + n)%n) * n + (i)%n];
        weightSum += w[3] * G[((i/n - 2 + n)%n) * n + (i + 1 + n)%n];
        weightSum += w[4] * G[((i/n - 2 + n)%n) * n + (i + 2 + n)%n];

        weightSum += w[5] * G[((i/n - 1 + n)%n) * n + (i - 2 + n)%n];
        weightSum += w[6] * G[((i/n - 1 + n)%n) * n + (i - 1 + n)%n];
        weightSum += w[7] * G[((i/n - 1 + n)%n) * n + (i)%n];
        weightSum += w[8] * G[((i/n - 1 + n)%n) * n + (i + 1 + n)%n];
        weightSum += w[9] * G[((i/n - 1 + n)%n) * n + (i + 2 + n)%n];

        weightSum += w[10] * G[((i/n + n)%n) * n + (i - 2 + n)%n];
        weightSum += w[11] * G[((i/n + n)%n) * n + (i - 1 + n)%n];
        // w[12] is not contributing anything. It's the current Atomic Spin.
        weightSum += w[13] * G[((i/n + n)%n) * n + (i + 1 + n)%n];
        weightSum += w[14] * G[((i/n + n)%n) * n + (i + 2 + n)%n];

        weightSum += w[15] * G[((i/n + 1 + n)%n) * n + (i - 2 + n)%n];
        weightSum += w[16] * G[((i/n + 1 + n)%n) * n + (i - 1 + n)%n];
        weightSum += w[17] * G[((i/n + 1 + n)%n) * n + (i)%n];
        weightSum += w[18] * G[((i/n + 1 + n)%n) * n + (i + 1 + n)%n];
        weightSum += w[19] * G[((i/n + 1 + n)%n) * n + (i + 2 + n)%n];

        weightSum += w[20] * G[((i/n + 2 + n)%n) * n + (i - 2 + n)%n];
        weightSum += w[21] * G[((i/n + 2 + n)%n) * n + (i - 1 + n)%n];
        weightSum += w[22] * G[((i/n + 2 + n)%n) * n + (i)%n];
        weightSum += w[23] * G[((i/n + 2 + n)%n) * n + (i + 1 + n)%n];
        weightSum += w[24] * G[((i/n + 2 + n)%n) * n + (i + 2 + n)%n];

        //! Can it be done more efficiently?
        if (weightSum > 0.0001)
            newG[i] = 1;
        else if (weightSum < -0.0001)
            newG[i] = -1;
        else
            newG[i] = G[i];
    }

    /*
    // Prints Atomic Spins Matrix.
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (G[i*n+j] == 1)
                printf("+");
            else
                printf("-");
        }
        printf("\n");
    }
    printf("\n\n");
    */

}

// This function will also be turned into a cuda kernel. Checks whether the new Atomic Spins Matrix is the same as the old one.
int check(int *G, int *newG, int n) {

    int same = 1;

    for (int i=0; i<n*n; i++)
	if (G[i] != newG[i]) {
	    same = 0;
            break;
	}

    return same;

}

void ising(int *G, double *w, int k, int n) {

    // Allocates memory for the new Atomic Spins Matrix.
    int *newG = (int *)malloc(n*n * sizeof(int));

    double timeAll = 0;
    double timeIteration = 0;	

    // Checks if function has to be iterated.
    for (int i=0; i<k; i++) {
        
        clock_t start = clock();

        spin(G, w, newG, n);
        
        clock_t end = clock();
        timeIteration = (double)(end-start)/CLOCKS_PER_SEC;
        timeAll += timeIteration;
        printf("Time taken in iteration %d: %f\n", i, timeIteration);

	// Checks if no further iterations are needed in case the new Atomic Spins Matrix is the same as the old one.
	if (check(G, newG, n))
	    break;

        // Atomix Spin Matrices' pointers swapping.
        int *temp = G;
        G = newG;
        newG = temp;
    }

    printf("\nAverage Time per iteration was: %f\n", timeAll/k);

    // Copies the result to the original G Atomic Spins Matrix. (G and newG pointers have been swapped swapped.)
    for (int i=0; i<n*n; i++)
        newG[i] = G[i];

    // Frees newG. (G and newG pointers have been swapped swapped.)
    free(G);

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

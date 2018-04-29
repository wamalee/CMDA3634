#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "cuda.h"
#include "functions.c" 

__device__ unsigned int devModprod(unsigned int a, unsigned int b, unsigned int p) {
  unsigned int za = a;
  unsigned int ab = 0;

  while (b>0) {
    if (b%2 == 1) ab = (ab + za) % p;
    za = (2 * za) % p;
    b /= 2;
  }
  return za;
}


__device__ unsigned int devModExp(unsigned int a, unsigned int b, unsigned int p) {
  unsigned int z = a;
  unsigned int aExpb = 1;

  while (b > 0) {
    if (b%2 == 1) aExpb = devModprod(aExpb, z, p);
    z = devModprod(z,z,p);
    b /= 2;
  }

  return aExpb;
}

__global__ void keyMaster(unsigned int p, unsigned int g, unsigned int h, unsigned int *deviceArray){
  unsigned int id = threadIdx.x + blockDim.x*blockIdx.x;
  
  if(id < p-1) {
    if (devModExp(g, id, p) == h) {
      deviceArray[0] = id;
    }
  }
}

int main (int argc, char **argv) {

  /* Part 2. Start this program by first copying the contents of the main function from 
     your completed decrypt.c main function. */

  /* Q4 Make the search for the secret key parallel on the GPU using CUDA. */

  //declare storage for an ElGamal cryptosytem
  unsigned int n, p, g, h, x;
  unsigned int Nints;
  unsigned int Nchars;

  //get the secret key from the user
  printf("Enter the secret key (0 if unknown): "); fflush(stdout);
  char stat = scanf("%u",&x);

  printf("Reading file.\n");

  /* Q3 Complete this function. Read in the public key data from public_key.txt
    and the cyphertexts from messages.txt. */

  FILE *file = fopen("public_key.txt", "r");
  fscanf(file, "%u\n%u\n%u\n%u\n", &n, &p, &g, &h);
  fclose(file);

  printf("Reading chars and ints\n");
 
  file = fopen("message.txt", "r");
  fscanf(file, "%u %u\n", &Nchars, &Nints);
 
  unsigned int *Z = (unsigned int *) malloc(Nints*sizeof(unsigned int));
  unsigned int *a = (unsigned int *) malloc(Nints*sizeof(unsigned int));

  printf("Reading %u message pairs\n", Nints);
  for (int i = 0; i < Nints; i++){
    fscanf(file, "%u %u\n", &Z[i], &a[i]);
    //printf("i is :%u\n", i);
  }
  fclose(file);
  
  printf("Searching for secret key\n");
  
  // set up cuda
  double startTime = clock();
 
  unsigned int Nthreads = 32;
  unsigned int *deviceArray, *hostArray;
   
  hostArray = (unsigned int *) malloc(Nthreads*sizeof(unsigned int));
  dim3 in(Nthreads,1,1);
  dim3 out((p+Nthreads-1)/Nthreads,1,1);
  cudaMalloc(&deviceArray,Nthreads*sizeof(unsigned int));

  keyMaster<<<out, in>>> (p,g,h,deviceArray);
  cudaDeviceSynchronize();
  cudaMemcpy(hostArray, deviceArray, Nthreads*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  x = hostArray[0];

  cudaFree(deviceArray);
  free(hostArray);  

  double endTime = clock();

  double totalTime = (endTime-startTime)/CLOCKS_PER_SEC;
  double work = (double) p;
  double throughput = work/totalTime;

    printf("Searching all keys took %g seconds, throughput was %g values tested per second.\n", totalTime, throughput);

  /* Q3 After finding the secret key, decrypt the message */

  unsigned char *message = (unsigned char *) malloc(1024*sizeof(unsigned char));
  
  ElGamalDecrypt(Z, a, Nints, p, x);
  convertZToString(Z, Nints, message, Nchars);

  printf("Here is the message: %s\n ", message);

  return 0;
}


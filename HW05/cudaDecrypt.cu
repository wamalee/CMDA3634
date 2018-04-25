#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "cuda.h"
#include "functions.c"

__global__ void keyMaster(int *g, int *p, int *h) {
  if (modExp(g, blockIdx.x+1, p) == h)
    dev_x = blockIdx.x + 1; 
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
  
  // GPU variables
  unsigned int *dev_g, *dev_p, *dev_h;
  cudaMalloc((void **) &dev_g, sizeof(unsigned int));
  cudaMalloc((void **) &dev_p, sizeof(unsigned int));
  cudaMalloc((void **) &dev_h, sizeof(unsigned int));  

  cudaMalloc((void **) &dev_x, sizeof(unsigned int));

  //Move data to GPU
  cudaMemcpy(dev_g, g, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_p, p, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_h, h, sizeof(unsigned int), cudaMemcpyHostToDevice);

  // find the secret key
  if (x==0 || modExp(g,x,p)!=h) {
    printf("Finding the secret key...\n");
    double startTime = clock();

    // Parallel Decrypt
    add<<< (p-1),1>>>(dev_g, dev_p, dev_h, dev_x);
    cudaMemcpy(dev_x, x, sizeof(unsigned int), cudaMemcpyDeviceToHost);    
    
    double endTime = clock();

    double totalTime = (endTime-startTime)/CLOCKS_PER_SEC;
    double work = (double) p;
    double throughput = work/totalTime;

    printf("Searching all keys took %g seconds, throughput was %g values tested per second.\n", totalTime, throughput);
  }

  /* Q3 After finding the secret key, decrypt the message */

  unsigned char *message = (unsigned char *) malloc(1024*sizeof(unsigned char));
  
  ElGamalDecrypt(Z, a, Nints, p, x);
  convertZToString(Z, Nints, message, Nchars);

  printf("Here is the message: %s\n ", message);


  return 0;
}


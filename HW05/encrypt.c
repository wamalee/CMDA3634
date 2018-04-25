#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "functions.h"

int main (int argc, char **argv) {

	//seed value for the randomizer 
  double seed = clock(); //this will make your program run differently everytime
  //double seed = 0; //uncomment this and your program will behave the same everytime it's run

  srand(seed);

  int bufferSize = 1024;
  unsigned char *message = (unsigned char *) malloc(bufferSize*sizeof(unsigned char));

  printf("Enter a message to encrypt: ");
  int stat = scanf (" %[^\n]%*c", message); //reads in a full line from terminal, including spaces

  //declare storage for an ElGamal cryptosytem
  unsigned int n, p, g, h;

  printf("Reading file.\n");

  /* Q2 Complete this function. Read in the public key data from public_key.txt,
    convert the string to elements of Z_p, encrypt them, and write the cyphertexts to 
    message.txt */
  
  FILE *file = fopen("public_key.txt", "r");
  fscanf(file, "%u\n%u\n%u\n%u\n", &n, &p, &g, &h);
  fclose(file);
  printf("Here are the read data\nn = %u\np = %u\ng = %u\nh = %u\n", n, p, g, h);
  
  // Set up variables
  unsigned int charsPerInt = (n-1)/8;
  unsigned int Nchars = mystrlen(message);
  unsigned int Nints = mystrlen(message)/charsPerInt;
  unsigned int *Z = (unsigned int *) malloc(Nints*sizeof(unsigned int));  
  unsigned int *a = (unsigned int *) malloc(Nints*sizeof(unsigned int));
  // Run Encryption

  padString(message, charsPerInt);
  convertStringToZ(message, Nchars, Z, Nints);
  ElGamalEncrypt(Z, a, Nints, p, g, h);

  // Print Pairs

  file = fopen("message.txt", "w");
  fprintf(file, "%u %u\n", Nchars, Nints);
  for (int i = 0; i < Nints; i++)
    fprintf(file, "%u %u\n", Z[i], a[i]);
  fclose(file);

  return 0;
}

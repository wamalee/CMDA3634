#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {

int rank;
char hostname[256];

MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
gethostname(hostname, 255);

printf("Hello! I am process %d on host %s\n", rank, hostname);

MPI_Finalize();
return 0;
}

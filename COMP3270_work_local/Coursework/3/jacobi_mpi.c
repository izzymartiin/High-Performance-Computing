#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "./system_gen.h"
#include <mpi.h>

/*
  Computes a 2-norm of a vector v.
*/
static double vector_sum(double *v, int n) {

  double sum = 0.0;
  for (int i = 0; i < n; i++)
    sum += v[i] * v[i];

  return sum;
}

/*
  Computes a 2-norm of Ax-b.
*/
static double residual_norm(double *A_local, double *x, double *b_local, int n, int n_local) {

  double sum_local = 0.0;

  for (int i = 0; i < n_local; i++) {
    double Ax_i = 0.0;

    for (int j = 0; j < n; j++)
      Ax_i += A_local[i*n + j] * x[j];

    double r = Ax_i - b_local[i];
    sum_local += r*r;
  }

  double sum_global = 0.0;
  MPI_Allreduce(&sum_local, &sum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return sqrt(sum_global);
}

/*
  Solves Ax=b via the Jacobi method with an initial solution in x.

  Inputs
    Max_iter: maximum number of Jacobi iterations to try.
    tolerance: relative residual error used to declare convergence.
    x: the initial solution vector of size n.

  Output:
    x: the solution to Ax=b.
*/
static double *jacobi(double *A_local, double *b_local, double *x,
                      int n, int n_local, int row_start, 
                      int *revcounts, int *displs, int max_iter, double tolerance, int rank) {

  double *x_new = calloc(n, sizeof(double));
  double *x_new_local = calloc(n_local, sizeof(double));

  double b_sumsq_local = vector_sum(b_local, n_local);
  double b_sumsq_global = 0.0;
  MPI_Allreduce(&b_sumsq_local, &b_sumsq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double b_norm = sqrt(b_sumsq_global);

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    for (int i = 0; i < n_local; i++) {
      int i_global = row_start + i;
      double sigma = 0.0;
      for (int j = 0; j < n; j++)
        if (j != i_global)
          sigma += A_local[i*n + j] * x[j];
      x_new_local[i] = (b_local[i] - sigma) / A_local[i*n + i_global];
    }

    MPI_Allgatherv(x_new_local, n_local, MPI_DOUBLE, x_new, revcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    double *tmp = x;
    x = x_new;
    x_new = tmp;

    // If the solution converged, stop iterating.
    double accuracy = residual_norm(A_local, x, b_local, n, n_local);
    if (accuracy / b_norm < tolerance) {
      if (rank == 0) {
        printf("Solved with error %e in %d iterations. \n",
             accuracy / b_norm, iter);
      }
     break;
    }
  }

  free(x_new);
  free(x_new_local);
  if (iter == max_iter && rank == 0) {
    printf("Solver did not converge in %d iterations. \n",
           max_iter);
  }
  return x;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int root = 0;

  if (argc < 3) {
    if (rank == root) {
    fprintf(stderr,
      "Please specify n the size of the square linear system and margin.\n");
    }
    MPI_Finalize();
    return 1;
  }

  int n = atoi(argv[1]);
  double margin = atof(argv[2]);

  int base = n/ size;
  int extra = n % size;
  //Work distribution
  int *row_counts = malloc(size * sizeof(int)); // rows per process
  int *row_displs = malloc(size *sizeof(int)); //first row of each process
  int *A_counts = malloc(size *sizeof(int)); // elements of A per process
  int *A_displs = malloc(size * sizeof(int)); //offset into A per process

  int offset = 0;
  for (int p = 0; p < size; p++) {
    row_counts[p] = base + (p < extra ? 1 : 0);
    row_displs[p] = offset;
    A_counts[p] = row_counts[p] * n;
    A_displs[p] = row_displs[p] *n;
    offset += row_counts[p];
  }

  int n_local = row_counts[rank];
  int row_start = row_displs[rank];

  //Allocation
  double *A_local = calloc((size_t) n_local * n, sizeof(double));
  double *b_local = calloc(n_local, sizeof(double));
  double *x = calloc(n, sizeof(double));

  //Only the root holds the full system
  double *A = NULL;
  double *b = NULL;
  if (rank == root) {
    A = calloc((size_t) n * n, sizeof(double));
    b = calloc(n, sizeof(double));
    if (!A || !b || !x) {
    fprintf(stderr, "Memory allocation for A, b, or x has failed. \n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  }

  if (!A_local || !b_local || !x) {
    fprintf(stderr, "Memory allocation for A, b, or x has failed. \n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  // Generate a linear system with diagonally dominant matrix A.
  //generate_system(A, b, n, 0, margin);
  if (rank == root) {
    generate_system(A, b, n, 0, margin);
  }

  MPI_Scatterv(A, A_counts, A_displs, MPI_DOUBLE, A_local, n_local * n, MPI_DOUBLE, root, MPI_COMM_WORLD);
  MPI_Scatterv(b, row_counts, row_displs, MPI_DOUBLE, b_local, n_local, MPI_DOUBLE, root, MPI_COMM_WORLD);

  double tolerance = 1e-10;
  int max_iter = 500;

  // Attempt to solve the linear system.
  x = jacobi(A_local, b_local, x, n, n_local, row_start, row_counts, row_displs, max_iter, tolerance, rank);

  double runTime = MPI_Wtime() - t0;
  if (rank == root) {
    printf("Total run time:     %f seconds\n", runTime);
  }

  free(A_local);
  free(b_local);
  free(x);
  free(row_counts);
  free(row_displs);
  free(A_counts);
  free(A_displs);
  if (rank == root) {
    free(A);
    free(b);
  }

  MPI_Finalize();
  return 0;
}

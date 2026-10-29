#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* Uniformly distributed random numbers in the range (0, 1). */
static double rand_uniform_open01(unsigned int *state) {
  return ((double)rand_r(state) + 1.0) / ((double)RAND_MAX + 2.0);
}

/* Normal distribution random numbers */
static double rand_normal_standard(unsigned int *state) {
  /* Box-Muller transform */
  const double PI = 3.14159265358979323846;
  double u1 = rand_uniform_open01(state);
  double u2 = rand_uniform_open01(state);
  return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

void simulateQueue(double T, double lambda_arrival, double mu_service, double C,
                   int runs) {

  const double dt = 0.01;   /* timestep [0.01 minutes = 0.6 seconds] */
  const int N = (int)(T / dt);
  const double sigma = 1.0; /* noise strength (stdev of 1 job/min) */

  double Qsum = 0.0;
  double Qdev = 0.0;

  double start = 0.0;
  double end = 0.0;

  int aloc_failed = 0;
  int threadid;
  int threads = omp_get_max_threads();

  start = omp_get_wtime();


#pragma omp parallel for \
  default(shared) \
  schedule(static) \
  reduction(+:Qsum) \
  reduction(+:Qdev) \

  for (int r = 0; r < runs; r++) {

    int threadid = omp_get_thread_num();
    
    /* Each run seeded with time, thread number times a large prime plus the run index */
    unsigned int seed = (unsigned int)(time(NULL) ^ (omp_get_thread_num() * 7919 + r));

    /* Allocate and initialise storage for Q, N timesteps. */
    double *Q = (double *)calloc((size_t)N, sizeof(double));

    if (Q == NULL) {
      #pragma omp atomic write
      aloc_failed = 1;
      continue;
    }

  /* Update the model for N timesteps. */
  for (int t = 0; t < N - 1; t++) {
    double q_min_c = (Q[t] < C) ? Q[t] : C;
    double drift = lambda_arrival - mu_service * q_min_c;
    double diffusion = sigma * sqrt(dt) * rand_normal_standard(&seed);

    Q[t + 1] = Q[t] + drift * dt + diffusion;
    if (Q[t + 1] < 0.0) {
      Q[t + 1] = 0.0;
    }
  }


  double mean = 0.0;
  for (int i = 0; i < N; i++) {
    mean += Q[i];
  }
  mean /= (double)N;

  /* Population variance. */
  double var = 0.0;
  for (int i = 0; i < N; i++) {
    double d = Q[i] - mean;
    var += d * d;
  }
  var /= (double)N;

  /* Compute sums of means and standard deviations. */
  Qsum += mean;
  Qdev += sqrt(var);

  free(Q);
  printf("Thread id: %d working on run %d.\n", threadid, r);
  }


  end = omp_get_wtime();
  

  if (aloc_failed == 1) {
    fprintf(stderr, "Memory allocation failed.\n");
    return;
  }

  FILE *fp = fopen("queue_parallel_benchmark_results.txt", "a+");
  if (fp == NULL) {
    perror("Error opening file");
  return;
  }

    // Check if empty
  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);

  if (size == 0) {
    fprintf(fp, "Threads Runtime\n");
  }
  fprintf(fp, "%d %f\n", threads, (end-start));

  printf("Queue statistics over %d runs\n", runs);
  printf("-------------------------------\n");
  printf("Mean queue length:  %f\n", Qsum / (double)runs);
  printf("Stdev queue length: %f\n", Qdev / (double)runs);
  printf("Mean occupancy:     %f%%\n", 100.0 * Qsum / ((double)runs * C));
  printf("Execution time: %f\n", end - start);
}

int main(int argc, char *argv[]) {
  int runs = 1;

  if (argc > 1) {
    runs = atoi(argv[1]);
    if (runs <= 0) {
      fprintf(stderr, "Invalid runs value. Using default runs = 1.\n");
      runs = 1;
    }
  }

  /* Set up random number generator seed */
  srand((unsigned int)time(NULL));

  simulateQueue(1200.0, 400.0 / 120.0, 0.2, 32.0, runs);

  return 0;
}

#include "RCppArmadillo.h"
#include <random>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <sys/time.h>

using namespace Rcpp;

struct matPair {
    arma::mat mat1;
    arma::mat mat2;
};

unsigned long int random_seed(void);
int test_gsl(void);
struct matPair draw_Z(arma::mat, arma::mat, arma::mat);


// [[Rcpp::depends(RcppArmadillo)]]


/*
// [[Rcpp::export]]
arma::Cube<unsigned int> draw_Z(arma::mat bulkExpr, arma::mat A, arma::mat W) {
    int N = A.n_rows;
    int M = bulkExpr.n_rows;
    int K = A.n_cols;
    arma::vec p_val = arma::vec(K);
    arma::mat AW = A * W;
    arma::Cube<unsigned int> Z = arma::Cube<unsigned int>(M, N, K);

    const gsl_rng_type * T;
    gsl_rng * r;
    unsigned int tmp[K];

    gsl_rng_env_setup();
  
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, random_seed());

    for (int j = 0; j < M; j++) {
        for (int i = 0; i < N; i++) {
            p_val = W.col(j) % A.row(i).t() / AW(i, j);
            gsl_ran_multinomial(r, K, bulkExpr(j, i), p_val.memptr(), tmp);
            for (int k = 0; k < K; k++) {
                Z(j, i, k) = tmp[k];
           }
        }
    }
    return Z;
}
*/


// [[Rcpp::export]]
List test_wrapper(arma::mat bulkExpr, arma::mat A, arma::mat W) {
    struct matPair Z = draw_Z(bulkExpr, A, W);
    return List::create(Named("Zjk") = Z.mat1,
                        Named("Zik") = Z.mat2);
}

struct matPair draw_Z(arma::mat bulkExpr, arma::mat A, arma::mat W) {
    int N = A.n_rows;
    int M = bulkExpr.n_rows;
    int K = A.n_cols;
    arma::vec p_val = arma::vec(K);
    arma::mat AW = A * W;
    arma::mat Zjk = arma::zeros<arma::mat>(M, K);
    arma::mat Zik = arma::zeros<arma::mat>(N, K);

    const gsl_rng_type * T;
    gsl_rng * r;
    unsigned int tmp[K];

    gsl_rng_env_setup();
  
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, random_seed());

    for (int j = 0; j < M; j++) {
        for (int i = 0; i < N; i++) {
            p_val = W.col(j) % A.row(i).t() / AW(i, j);
            gsl_ran_multinomial(r, K, bulkExpr(j, i), p_val.memptr(), tmp);
            for (int k = 0; k < K; k++) {
                Zjk(j, k) += tmp[k];
                Zik(i, k) += tmp[k];
           }
        }
    }
    struct matPair Z = {Zjk, Zik};

    return Z;
}


// arma::mat draw_W(arma::vec alpha, arma::Cube<unsigned int> Z) {
//     return ;
// }

unsigned long int
random_seed(void)
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (tv.tv_sec + tv.tv_usec);
}

int
test_gsl (void)
{
    const gsl_rng_type * T;
    gsl_rng * r;
  
    int i, n = 10;
    double mu = 3.0;
  
    /* create a generator chosen by the 
       environment variable GSL_RNG_TYPE */
  
    gsl_rng_env_setup();
  
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, random_seed());
  
    /* print n random variates chosen from 
       the poisson distribution with mean 
       parameter mu */
  
    for (i = 0; i < n; i++) {
        unsigned int k = gsl_ran_poisson (r, mu);
        printf (" %u", k);}
  
    printf ("\n");
    gsl_rng_free (r);
    return 0;
}

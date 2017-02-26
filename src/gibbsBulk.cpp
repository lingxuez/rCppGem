#include "RCppArmadillo.h"

// #ifndef GIBBS_BULK_H
// #define GIBBS_BULK_H
#include "gibbsBulk.h"
// #endif


using namespace Rcpp;

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
arma::Mat<int> test_draw_Z(const arma::Mat<int> & bulkExpr, const arma::mat & A, const arma::mat & W,
            arma::Mat<int> & Zik, arma::Mat<int> & Zjk)
{
    draw_Z(bulkExpr, A, W, Zik, Zjk);
    return Zjk;
}

void draw_Z(const arma::Mat<int> & bulkExpr, const arma::mat & A, const arma::mat & W,
            arma::Mat<int> & Zik, arma::Mat<int> & Zjk)
{
    int N = A.n_rows;
    int M = bulkExpr.n_rows;
    int K = A.n_cols;
    arma::vec p_val = arma::vec(K);
    arma::mat AW = A * W;
    Zik.zeros();
    Zjk.zeros();

    // const gsl_rng_type * T;
    // gsl_rng * r;
    arma::Col<int> tmp = arma::Col<int>(K);
    tmp.zeros();

    // gsl_rng_env_setup();
  
    // T = gsl_rng_default;
    // r = gsl_rng_alloc (T);
    // gsl_rng_set(r, random_seed());

    for (int j = 0; j < M; j++) {
        for (int i = 0; i < N; i++) {
            p_val = W.col(j) % A.row(i).t() / AW(i, j);
            rmultinom(bulkExpr(j, i), p_val.memptr(), K, tmp.memptr());
            // gsl_ran_multinomial(r, K, bulkExpr(j, i), p_val.memptr(), tmp);
            for (int k = 0; k < K; k++) {
                Zjk(j, k) += tmp[k];
                Zik(i, k) += tmp[k];
           }
        }
    }
}


void draw_W(const arma::vec & alpha, const arma::Mat<int> & Zjk, arma::mat & W) {
    int M = Zjk.n_rows;
    int K = Zjk.n_cols;
    arma::vec tmp = arma::vec(K);
    for (int j = 0; j < M; j++) {
        for (int k = 0; k < K; k++) {
            tmp(k) = rgamma(1, alpha(k) + Zjk(j, k), 1)(0);
        }
        W.col(j) = tmp / sum(tmp);
    }
}

// unsigned long int
// random_seed(void)
// {
//   struct timeval tv;
//   gettimeofday(&tv, 0);
//   return (tv.tv_sec + tv.tv_usec);
// }
// 
// int
// test_gsl (void)
// {
//     const gsl_rng_type * T;
//     gsl_rng * r;
//   
//     int i, n = 10;
//     double mu = 3.0;
//   
//     /* create a generator chosen by the 
//        environment variable GSL_RNG_TYPE */
//   
//     gsl_rng_env_setup();
//   
//     T = gsl_rng_default;
//     r = gsl_rng_alloc (T);
//     gsl_rng_set(r, random_seed());
//   
//     /* print n random variates chosen from 
//        the poisson distribution with mean 
//        parameter mu */
//   
//     for (i = 0; i < n; i++) {
//         unsigned int k = gsl_ran_poisson (r, mu);
//         printf (" %u", k);}
//   
//     printf ("\n");
//     gsl_rng_free (r);
//     return 0;
// }


void update_SuffStat_BK(const arma::Mat<int> & Zik, const arma::Mat<int> & Zjk, const arma::mat & W,
                        arma::mat & exp_Zik, arma::mat & exp_Zjk, arma::mat & exp_W, arma::mat & exp_logW, int n_samples)
{
    exp_Zik += arma::conv_to<arma::mat>::from(Zik) / n_samples;
    exp_Zjk += arma::conv_to<arma::mat>::from(Zjk) / n_samples;
    exp_W += W / n_samples;
    exp_logW += arma::log(W) / n_samples;
}




void gibbs_BK(int K, int M, int N, const arma::Mat<int> & i_markers,
              const arma::Mat<int> & bulkExpr, const arma::vec & alpha,
              const arma::mat & A, arma::mat & exp_Zik, arma::mat & exp_Zjk,
              arma::mat & exp_W, arma::mat & exp_logW,
              int burn_in, int n_samples, int thin)
{
    int i, k;
    arma::mat W = arma::mat(K, M);
    arma::Mat<int> Zik = arma::Mat<int>(N, K);
    arma::Mat<int> Zjk = arma::Mat<int>(M, K, arma::fill::zeros);


    // Initialize Zjk according to Marker Genes

    for (int i_row = 0; i_row < i_markers.n_rows; i_row++) {
        i = i_markers(i_row, 0);
        k = i_markers(i_row, 1);
        for (int j = 0; j < M; j++) {
            Zjk(j, k) = bulkExpr(j, i);
        }
    }

    

    // Gibbs Sampling: Burn In
    for (int i_iter = 0; i_iter < burn_in; i_iter++) {
        draw_W(alpha, Zjk, W);
        draw_Z(bulkExpr, A, W, Zik, Zjk);
    }

    // Gibbs Sampling
    for (int i_iter = 0; i_iter < n_samples * thin; i_iter++) {
        draw_W(alpha, Zjk, W);
        draw_Z(bulkExpr, A, W, Zik, Zjk);
        if (i_iter % thin == 0) {
            update_SuffStat_BK(Zik, Zjk, W, exp_Zik, exp_Zjk, exp_W, exp_logW, n_samples);
        }
    }
}

#include "RCppArmadillo.h"

// #ifndef GIBBS_BULK_H
// #define GIBBS_BULK_H
#include "gibbsBulk.h"
#include "gibbsSC.h"
// #endif



using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

// std::vector< std::vector<int> > 

std::vector< std::vector<int> > get_i_types(const arma::Col<int> & G, int K)
{
    std::vector<int> counts(K);

    for (int i = 0; i < G.n_elem; i++) {
        counts[G(i)]++;
    }

    std::vector<std::vector<int> > i_types(K);

    for (int i = 0; i < G.n_elem; i++) {
        i_types[G(i)].push_back(i);
    }
    
    // for (int k = 0; k < K; k++) {
    //     std::cout << "k = " << k << "\n";
    //     for (int i = 0; i < i_types[k].size(); i++) {
    //         std::cout << "val: " << i_types[k][i] << "\t";
    //     }
    //     std::cout << "\n";
    // }

    return i_types;
}


// [[Rcpp::export]]
void gem(double EM_CONV, double MLE_CONV, int EM_MAX_ITER, int MLE_MAX_ITER, 
         const arma::Mat<int> & bulkExpr, const arma::Mat<int> & scExpr, 
         const arma::Col<int> & G, int K, const arma::Mat<int> & i_markers,
         arma::vec & alpha, arma::vec & p_kappa, arma::vec & p_tau, 
         const arma::Mat<int> & i_zeros, const arma::vec & SCrd,
         bool has_BK, bool has_SC,
         int M, int L, int N,
         int burn_in, int n_samples, int thin)
{
    // int M = bulkExpr.n_rows;
    // int L = scExpr.n_rows;
    // int N = scExpr.n_cols;

    // Compute number of cells in each type
    std::vector<int> counts(K);
    if (has_SC) {
        for (int i = 0; i < G.n_elem; i++) {
            counts[G(i)]++;
        }
    }

    // Initialize A according to normalized single cell mean expression
    arma::mat A = arma::mat(N, K, arma::fill::zeros);
    for (int l = 0; l < L; l++) {
        A.col(G(l)) = A.col(G(l)) + arma::conv_to<arma::mat>::from(scExpr).row(l).t() / arma::sum(scExpr.row(l));
    }
    for (int k = 0; k < K; k++) {
        double col_sum = arma::sum(A.col(k));
        if (col_sum == 0) {
            A.col(k).fill(1.0 / N);
        } else {
            A.col(k) /= col_sum;
        }
    }

    // std::cout << "A:\n" << A << "\n";


    // Initialize sufficient statistics for bulk
    arma::mat exp_Zik = arma::mat(N, K, arma::fill::zeros);
    arma::mat exp_Zjk = arma::mat(M, K, arma::fill::zeros);
    arma::mat exp_W = arma::mat(K, M, arma::fill::zeros);
    arma::mat exp_logW = arma::mat(K, M, arma::fill::zeros);
    

    // Initialize sufficient statistics for single cell
    arma::mat exp_S = arma::mat(L, N, arma::fill::zeros);
    arma::vec exp_kappa = arma::vec(L, arma::fill::zeros);
    arma::vec exp_tau = arma::vec(L, arma::fill::zeros);
    arma::vec exp_kappa_sq = arma::vec(L, arma::fill::zeros);
    arma::vec exp_tau_sq = arma::vec(L, arma::fill::zeros);
    arma::mat coeff_A = arma::mat(N, K, arma::fill::zeros);
    arma::mat coeff_A_sq = arma::mat(N, K, arma::fill::zeros);

    // gem

    double converge = EM_CONV + 1;
    int n_iter = 0;
    double exp_elbo_const = 0;

    while (converge > EM_CONV && n_iter < EM_MAX_ITER) {
        // E Step: Gibbs Sampling
        if (has_BK) {
            gibbs_BK(K, M, N, i_markers, bulkExpr, alpha, A, exp_Zik, exp_Zjk, exp_W, exp_logW,
                     burn_in, n_samples, thin);
        }
        if (has_SC) {
            exp_elbo_const = gibbs_SC(L, N, K, A, G, p_kappa, p_tau, i_zeros, exp_S, exp_kappa, exp_tau, exp_kappa_sq,
                     exp_tau_sq, coeff_A, coeff_A_sq, SCrd, burn_in, n_samples, thin);
        }

        // M Step:
        

        n_iter++;
    }

    
    
}



// void gibbs()
// {
// 
// }

#include "RcppArmadillo.h"
#include "RNG.h"
#include "PolyaGamma.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

void draw_S(const arma::Mat<int> & i_zeros, const arma::mat & A, const arma::Col<int> & G, 
            arma::Mat<int> & S, const arma::mat & psi, arma::vec & sumAS, const arma::vec & SCrd);
void draw_omega(arma::mat & omega, arma::mat & psi,
                const arma::mat & A, const arma::vec & kappa, const arma::vec & tau,
                const arma::Col<int> & G);
void draw_kappa_tau(const arma::mat & A, const arma::Col<int> & G, const arma::mat & omega, const arma::vec & p_kappa,
                    const arma::vec & p_tau, const arma::vec & sumAS, arma::vec & kappa, arma::vec & tau, const arma::Mat<int> & S);


void draw_omega(arma::mat & omega, arma::mat & psi,
                const arma::mat & A, const arma::vec & kappa, const arma::vec & tau,
                const arma::Col<int> & G)
{
    RNG r;
    PolyaGamma pg;

    int N = A.n_rows;
    int L = G.n_elem;
    // arma::mat omega = arma::mat(N, L);
    // arma::mat psi = arma::mat(N, L);

    GetRNGstate();

    for (int l = 0; l < L; l++) {
        for (int i = 0; i < N; i++) {
            psi(l, i) = kappa(l) + tau(l) * A(i, G(l));
            omega(l, i) = pg.draw(1, psi(l, i), r);
        }
    }
    
    PutRNGstate();
}



// [[Rcpp::export]]
arma::mat test_draw_omega(arma::mat omega, arma::mat psi,
                const arma::mat A, const arma::vec kappa, const arma::vec tau,
                const arma::Col<int> G)
{
    draw_omega(omega, psi, A, kappa, tau, G);
    return psi;
}

// [[Rcpp::export]]
arma::Mat<int> test_draw_S(const arma::Mat<int> & i_zeros, const arma::mat & A, const arma::Col<int> & G, 
            arma::Mat<int> & S, const arma::mat & psi, arma::vec & sumAS, const arma::vec & SCrd)
{
    draw_S(i_zeros, A, G, S, psi, sumAS, SCrd);
    return S;
}


void draw_S(const arma::Mat<int> & i_zeros, const arma::mat & A, const arma::Col<int> & G, 
            arma::Mat<int> & S, const arma::mat & psi, arma::vec & sumAS, const arma::vec & SCrd)
{
    int l, i;
    double A_cur, other, b;
    for (int i_row = 0; i_row < i_zeros.n_rows; i_row ++) {
        l = i_zeros(i_row, 0);
        i = i_zeros(i_row, 1);
        A_cur = A(i, G(l));
        other = sumAS(l) - A_cur * S(l, i);
        if (other == 0) {
            b = 1 / (1 + exp(-psi(l, i)));
        } else {
            b = 1 / (1 + exp(-psi(l, i) + SCrd(l) * log(1 + A_cur / other)));
        }
        S(l, i) = rbinom(1, 1, b)(0);
        sumAS(l) = other + A_cur * S(l, i);
    }
}


// [[Rcpp::export]]
arma::vec test_draw_kappa_tau(const arma::mat & A, const arma::Col<int> & G, const arma::mat & omega, const arma::vec & p_kappa,
                    const arma::vec & p_tau, const arma::vec & sumAS, arma::vec & kappa, arma::vec & tau, const arma::Mat<int> & S) 
{
    draw_kappa_tau(A, G, omega, p_kappa, p_tau, sumAS, kappa, tau, S);
    return kappa;
}



void draw_kappa_tau(const arma::mat & A, const arma::Col<int> & G, const arma::mat & omega, const arma::vec & p_kappa,
                    const arma::vec & p_tau, const arma::vec & sumAS, arma::vec & kappa, arma::vec & tau, const arma::Mat<int> & S) 
{
    int L = G.n_elem;
    int N = A.n_rows;
    arma::mat PP = arma::mat(2, 2);
    arma::vec bP = arma::vec(2);
    arma::vec mP = arma::vec(2);
    arma::mat cov_sqrt = arma::mat(2, 2);
    arma::vec z = arma::vec(2);
    for (int l = 0; l < L; l++) {
        PP.zeros();
        bP.zeros();
        for (int i = 0; i < N; i++) {
            PP(0, 1) += omega(l, i) * A(i, G(l));
            PP(0, 0) += omega(l, i);
            PP(1, 1) += omega(l, i) * A(i, G(l)) * A(i, G(l));
            bP(0) += S(l, i);
        }
        PP(1, 0) = PP(0, 1);
        PP(0, 0) += p_kappa(1);
        PP(1, 1) += p_tau(1);
        bP(0) += p_kappa(0) * p_kappa(1) - (double) N / 2;
        bP(1) += sumAS(l) - 0.5 + p_tau(0) * p_tau(1);
        mP = arma::solve(PP, bP);

        cov_sqrt = arma::sqrtmat_sympd(PP.i());

        z = rnorm(2);

        mP += cov_sqrt * z;

        kappa(l) = mP(0);
        tau(l) = mP(1);
    }
}

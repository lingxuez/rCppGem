#include "RcppArmadillo.h"
#include "RNG.h"
#include "PolyaGamma.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

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
            psi(i, l) = kappa(l) + tau(l) * A(i, G(l));
            omega(i, l) = pg.draw(1, psi(i, l), r);
        }
    }
    
    PutRNGstate();
}


// [[Rcpp::export]]
arma::mat test_draw_omega(arma::mat omega, arma::mat psi,
                const arma::mat A, const arma::vec kappa, const arma::vec tau,
                const arma::Col<int> G) {
    draw_omega(omega, psi, A, kappa, tau, G);
    return omega;
}

// arma::mat draw_S(arma::Mat<int> i_zeros, arma::mat A, arma::Col<int> G, 

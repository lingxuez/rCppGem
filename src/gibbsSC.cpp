#include "RcppArmadillo.h"
#include "RNG.h"
#include "PolyaGamma.h"
#include "gibbsSC.h"

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
    return omega;
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
    for (int i_row = 0; i_row < i_zeros.n_rows; i_row++) {
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

double update_SuffStat_SC(const arma::Mat<int> & S, const arma::vec & kappa, const arma::vec & tau, const arma::mat & omega, 
                   int n_samples, arma::mat & exp_S, arma::vec & exp_kappa, arma::vec & exp_tau,
                   arma::vec & exp_kappa_sq, arma::vec & exp_tau_sq, arma::mat & coeff_A, arma::mat & coeff_A_sq,
                   const arma::Col<int> & G)
{

    // std::cout << "n samples: \n" << n_samples << "\n";

    exp_S += arma::conv_to<arma::mat>::from(S) / n_samples;
    exp_kappa += kappa / n_samples;
    exp_tau += tau / n_samples;
    exp_kappa_sq += arma::square(kappa) / n_samples;
    exp_tau_sq += arma::square(tau) / n_samples;

    // std::cout << "exp_S: \n" << exp_S << "\n";
    // std::cout << "exp_kappa: \n" << exp_kappa << "\n";
    // std::cout << "exp_tau: \n" << exp_tau << "\n";
    // std::cout << "exp_kappa_sq: \n" << exp_kappa_sq << "\n";
    // std::cout << "exp_tau_sq: \n" << exp_tau_sq << "\n";

    // coeff_A.zeros();
    // coeff_A_sq.zeros();

    double exp_elbo_const = 0;

    for (int i = 0; i < coeff_A.n_rows; i++) {
        for (int l = 0; l < S.n_rows; l++) {
            coeff_A(i, G(l)) += ((S(l, i) - 0.5) * tau(l) - omega(l, i) * tau(l) * kappa(l)) / n_samples;
            coeff_A_sq(i, G(l)) -= (omega(l, i) * tau(l) * tau(l) / 2) / n_samples;
            exp_elbo_const += (S(l, i) - 0.5) * kappa(l) - omega(l, i) * kappa(l) * kappa(l) / 2;
        }
    }

    exp_elbo_const /= n_samples;

    // std::cout << "coeff_A: \n" << coeff_A << "\n";
    // std::cout << "coeff_A_sq: \n" << coeff_A_sq << "\n";

    return exp_elbo_const;
}





double gibbs_SC(int L, int N, int K, const arma::mat & A, const arma::Col<int> & G, 
              const arma::vec & p_kappa, const arma::vec & p_tau, 
              const arma::Mat<int> & i_zeros,
              arma::mat & exp_S, arma::vec & exp_kappa, arma::vec & exp_tau,
              arma::vec & exp_kappa_sq, arma::vec & exp_tau_sq,
              arma::mat & coeff_A, arma::mat & coeff_A_sq,
              const arma::vec & SCrd,
              int burn_in, int n_samples, int thin)
{
    //Initialize
    arma::vec kappa = arma::vec(L, arma::fill::ones);
    arma::vec tau = arma::vec(L, arma::fill::ones);
    kappa *= p_kappa(0);
    tau *= p_tau(0);
    arma::Mat<int> S = arma::Mat<int>(L, N, arma::fill::ones);
    arma::mat psi = arma::mat(L, N);
    arma::mat omega = arma::mat(L, N);
    arma::vec sumAS = arma::vec(L);

    double exp_elbo_const = 0;

    for (int l, i, i_row = 0; i_row < i_zeros.n_rows; i_row++) {
        l = i_zeros(i_row, 0);
        i = i_zeros(i_row, 1);
        S(l, i) = rbinom(1, 1, 0.5)(0);
    }

    for (int l = 0; l < L; l++) {
        sumAS(l) = arma::sum(A.col(G(l)) % S.row(l).t());
    }

    //Gibbs Sampling: Burn In
    for (int i_iter = 0; i_iter < burn_in; i_iter++) {
        draw_omega(omega, psi, A, kappa, tau, G);
        draw_S(i_zeros, A, G, S, psi, sumAS, SCrd);
        draw_kappa_tau(A, G, omega, p_kappa, p_tau, sumAS, kappa, tau, S);
    }

    //Gibbs Sampling
    for (int i_iter = 0; i_iter < n_samples * thin; i_iter++) {
        draw_omega(omega, psi, A, kappa, tau, G);
        draw_S(i_zeros, A, G, S, psi, sumAS, SCrd);
        draw_kappa_tau(A, G, omega, p_kappa, p_tau, sumAS, kappa, tau, S);
        if (i_iter % thin == 0)
        {
            exp_elbo_const += update_SuffStat_SC(S, kappa, tau, omega, n_samples, exp_S, exp_kappa, exp_tau,
                               exp_kappa_sq, exp_tau_sq, coeff_A, coeff_A_sq, G);
        }
    }
    
    return exp_elbo_const;
}

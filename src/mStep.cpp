#include "RCppArmadillo.h"


using namespace Rcpp;

void simplex_proj(arma::mat & A, int k, double min_A);

void opt_Ak_Bulk(arma::mat & A, int k, const arma::mat & exp_Zik, double min_A);

void opt_Ak_full(arma::mat & A, int k, double MLE_CONV, int MLE_MAX_ITER,
                 double init_stepsize, double min_A, 
                 const arma::mat & gd_coeffA, 
                 const arma::mat & gd_coeffAinv,
                 const arma::Col<int> & G, const arma::mat & coeffA,
                 const arma::mat & exp_S, const arma::vec & SCrd,
                 arma::mat & gd_coeffConst, arma::vec & u);

void opt_kappa_tau(const arma::vec & exp_kappa, const arma::vec & exp_kappa_sq,
                   const arma::vec & exp_tau, const arma::vec & exp_tau_sq,
                   arma::vec & p_kappa, arma::vec & p_tau);

arma::vec get_grad_Ak(const arma::vec & Ak,
                      const arma::mat & gd_coeffA, 
                      const arma::mat & gd_coeffAinv,
                      const arma::mat & gd_coeffConst, 
                      int k);

double get_obj_Ak(const arma::vec & Ak,
                  const arma::mat & gd_coeffA, 
                  const arma::mat & gd_coeffAinv,
                  const arma::mat & gd_coeffConst, 
                  int k);

double compute_elbo(double exp_elbo_const, const arma::mat & gd_coeffAinv, const arma::mat & A,
                    const arma::mat & gd_coeffA, const arma::mat & gd_coeffConst, 
                    const arma::vec & SCrd, const arma::vec & u,
                    // const arma::vec & p_kappa, 
                    // const arma::vec & p_tau, const arma::vec & exp_kappa_sq, const arma::vec & exp_kappa,
                    // const arma::vec & exp_tau_sq, const arma::vec & exp_tau,
                    bool has_BK, bool has_SC
                    );
// void opt_Ak_full(arma::mat & A, int k);

// [[Rcpp::depends(RcppArmadillo)]]
void opt_kappa_tau(const arma::vec & exp_kappa, const arma::vec & exp_kappa_sq,
                   const arma::vec & exp_tau, const arma::vec & exp_tau_sq,
                   arma::vec & p_kappa, arma::vec & p_tau)
{
    p_kappa(0) = arma::mean(exp_kappa);
    p_tau(0) = arma::mean(exp_tau);
    p_kappa(1) = arma::mean(exp_kappa_sq) - p_kappa(0) * p_kappa(0);
    p_tau(1) = arma::mean(exp_tau_sq) - p_tau(0) * p_tau(0);

}


// [[Rcpp::export]]
void opt_MLE(std::vector<int> counts, arma::mat & A, const arma::mat & exp_Zik, double min_A,
           double MLE_CONV, int MLE_MAX_ITER, double init_stepsize, const arma::Col<int> & G,
           const arma::vec & SCrd, const arma::mat & exp_Zjk, const arma::mat & exp_logW,
           const arma::mat & coeff_A, const arma::mat & coeff_A_sq, double exp_elbo_const,
           const arma::Mat<int> & scExpr, const arma::mat & exp_S, bool has_BK, bool has_SC)
{

    int N = A.n_rows;
    int K = A.n_cols;
    int L = scExpr.n_rows;
    arma::mat gd_coeffAinv = exp_Zik;
    arma::mat gd_coeffA = coeff_A_sq * 2;
    arma::mat gd_coeffConst = coeff_A;

    std::cout << "coeff A: \n" << coeff_A << "\n";
    gd_coeffConst(0, 0) = 1000;
    std::cout << "coeff A: \n" << coeff_A << "\n";
    std::cout << "gd_coeffConst address: " << & gd_coeffConst << "\n";
    std::cout << "coeff_A address: " << & coeff_A << "\n";

    double elbo_const = exp_elbo_const;
    arma::vec u = arma::vec(L);

    // compute coefficients for gradients of A
    // for single cell
    if (has_SC) {
        for (int l = 0; l < L; l++) {
            gd_coeffAinv.col(G(l)) += (scExpr.row(l) % exp_S.row(l)).t();
        }

        for (int l = 0; l < L; l++) {
            u(l) = arma::dot(exp_S.row(l), A.col(G(l)));
            gd_coeffConst.col(G(l)) -= exp_S.row(l).t() * SCrd(l) / u(l);
        }
    }

    std::cout << "U vector: \n" << u << "\n";

    // optimize A col by col
    for (int k = 0; k < K; k++) {
        if (counts[k] == 0) {
            // Only bulk sample
            opt_Ak_Bulk(A, k, exp_Zik, min_A);
        } else {
            // Both bulk and single cell
            opt_Ak_full(A, k, MLE_CONV, MLE_MAX_ITER, init_stepsize, min_A, gd_coeffA,
                        gd_coeffAinv, G, coeff_A, exp_S, SCrd, gd_coeffConst, u);
        }
    }
    // std::cout << "Optimized A:\n" << A << "\n";
}

// project the k-th column of A to simplex
void simplex_proj(arma::mat & A, int k, double min_A) 
{
    int N = A.n_rows;
    arma::vec y_sorted = arma::sort(A.col(k), "descend");
    arma::vec y_cumsum = arma::cumsum(y_sorted);
    arma::vec proj_Ak = arma::vec(N);
    int i_best = -1;
    double rou, rou_best = -1;

    for (int i = 0; i < N; i++) {
        rou = y_sorted(i) + (1 - (N - i - 1) * min_A - y_cumsum(i)) / (double) (i + 1);
        if (rou > min_A) {
            i_best = i;
            rou_best = rou;
        }
    }
    double lambda = rou_best - y_sorted(i_best);
    
    for (int i = 0; i < N; i++) {
        A(i, k) = std::max(A(i, k) + lambda, min_A);
    }
}

void opt_Ak_Bulk(arma::mat & A, int k, const arma::mat & exp_Zik, double min_A)
{
    A.col(k) = exp_Zik.col(k) / arma::sum(exp_Zik.col(k));
    simplex_proj(A, k, min_A);
}


arma::vec get_grad_Ak(const arma::vec & Ak,
                      const arma::mat & gd_coeffA, 
                      const arma::mat & gd_coeffAinv,
                      const arma::mat & gd_coeffConst, 
                      int k)
{
    int N = Ak.n_elem;
    arma::vec grad_Ak = arma::vec(N);
    grad_Ak = gd_coeffAinv.col(k) / Ak + gd_coeffA.col(k) % Ak + gd_coeffConst.col(k);
    grad_Ak /= N;
    return grad_Ak;
}


double get_obj_Ak(const arma::vec & Ak,
                  const arma::mat & gd_coeffA, 
                  const arma::mat & gd_coeffAinv,
                  const arma::mat & gd_coeffConst, 
                  int k)
{
    double obj_Ak = arma::sum(gd_coeffAinv.col(k) % arma::log(Ak));
    obj_Ak += arma::sum(gd_coeffA.col(k) % Ak % Ak) / 2 + arma::sum(gd_coeffConst.col(k) % Ak);
    return obj_Ak;
}


// [[Rcpp::export]]
void opt_Ak_full(arma::mat & A, int k, double MLE_CONV, int MLE_MAX_ITER,
                 double init_stepsize, double min_A, 
                 const arma::mat & gd_coeffA, 
                 const arma::mat & gd_coeffAinv,
                 const arma::Col<int> & G, const arma::mat & coeffA,
                 const arma::mat & exp_S, const arma::vec & SCrd,
                 arma::mat & gd_coeffConst, arma::vec & u)
{
    int N = A.n_rows;
    arma::mat old_Ak = A.col(k);
    arma::mat new_Ak = arma::mat(N, 1);
    arma::vec old_grad = arma::vec(N);
    arma::vec new_grad = arma::vec(N);
    arma::vec old_Gt = arma::vec(N);
    double old_obj, new_obj, stepsize, tmp_obj;

    double converge = MLE_CONV + 1;
    double tmp_elbo;
    int n_iter = 0;
    std::cout << "optimizing " << k << "\n";

    while (converge > MLE_CONV && n_iter < MLE_MAX_ITER) {
        // Backtracking
        stepsize = init_stepsize;
        old_grad = get_grad_Ak(old_Ak, gd_coeffA, gd_coeffAinv, gd_coeffConst, k);
        old_obj = get_obj_Ak(old_Ak, gd_coeffA, gd_coeffAinv, gd_coeffConst, k);
        new_Ak = old_Ak + stepsize * old_grad;
        simplex_proj(new_Ak, 0, min_A);

        A.col(k) = old_Ak;
        tmp_elbo = compute_elbo(0, gd_coeffAinv, A, gd_coeffA, gd_coeffConst, SCrd, u, true, true);
        std::cout << "old temp elbo: \n" << tmp_elbo << "\n";

        old_Gt = (old_Ak - new_Ak) / stepsize;
        new_obj = get_obj_Ak(new_Ak, gd_coeffA, gd_coeffAinv, gd_coeffConst, k);

        tmp_obj = old_obj - stepsize * 0.5 * arma::sum(old_Gt % old_Gt) - stepsize * arma::dot(old_Ak.t(), old_Gt);

        // std::cout << "outside while: tmp obj:\n" << tmp_obj << "\n";
        // std::cout << "outside while: new obj:\n" << new_obj << "\n";

        while (new_obj < tmp_obj) {
            // Shrink stepsize by half
            stepsize /= 2;
            new_Ak = old_Ak + stepsize * old_grad;

            simplex_proj(new_Ak, 0, min_A);
            old_Gt = (old_Ak - new_Ak) / stepsize;
            new_obj = get_obj_Ak(new_Ak, gd_coeffA, gd_coeffAinv, gd_coeffConst, k);

            tmp_obj = old_obj - stepsize * 0.5 * arma::sum(old_Gt % old_Gt) - stepsize * arma::dot(old_Ak.t(), old_Gt);

            // std::cout << "New obj:\n" << new_obj << "\n";
        }

        old_Ak = new_Ak;
        std::cout << "old AK address" << & old_Ak << "\n";
        std::cout << "new AK address" << & new_Ak << "\n";

        // debug
        A.col(k) = new_Ak;
        tmp_elbo = compute_elbo(0, gd_coeffAinv, A, gd_coeffA, gd_coeffConst, SCrd, u, true, true);
        std::cout << "new temp elbo: \n" << tmp_elbo << "\n";

        // Update auxiliary u
        gd_coeffConst.col(k) = coeffA.col(k);
        std::cout << "gd_coeffConst col k address" << gd_coeffConst.colptr(k) << "\n";
        std::cout << "coeffA col k address" << coeffA.colptr(k) << "\n";
        
        int L = exp_S.n_rows;
        for (int l = 0; l < L; l++) {
            if (G(l) == k) {
                u(l) = arma::dot(exp_S.row(l), new_Ak);
                gd_coeffConst.col(k) -= exp_S.row(l).t() * SCrd(l) / u(l);
            }
        }
        
        n_iter++;
    }
    // std::cout << "New Ak: \n" << new_Ak << "\n";
    // std::cout << "Old grad: \n" << old_grad << "\n";
}



double compute_elbo(double exp_elbo_const, const arma::mat & gd_coeffAinv, const arma::mat & A,
                    const arma::mat & gd_coeffA, const arma::mat & gd_coeffConst, 
                    const arma::vec & SCrd, const arma::vec & u,
                    // const arma::vec & p_kappa, 
                    // const arma::vec & p_tau, const arma::vec & exp_kappa_sq, const arma::vec & exp_kappa,
                    // const arma::vec & exp_tau_sq, const arma::vec & exp_tau,
                    bool has_BK, bool has_SC
                    )
{
    int N = A.n_rows;
    double elbo = exp_elbo_const;
    elbo += arma::accu(gd_coeffAinv % arma::log(A));

    if (has_SC) {
        int L = SCrd.n_elem;
        elbo += arma::accu(gd_coeffA % A % A) / 2;
        elbo += arma::accu(gd_coeffConst % A);
        elbo -= arma::accu(SCrd % arma::log(u));
        // elbo += arma::log(p_kappa(1) * p_tau(1)) * L / 2;
        // elbo -= arma::sum(exp_kappa_sq) - 2 * p_kappa(0) * amra::sum(exp_kappa) +
        //         L * p_kappa(0) * p_kappa(0) * p_kappa(1) / 2;
        // elbo -= arma::sum(exp_tau_sq) - 2 * p_tau(0) * amra::sum(exp_tau) +
        //         L * p_tau(0) * p_tau(0) * p_tau(1) / 2;
    }

    elbo /= N;

    return elbo;
}

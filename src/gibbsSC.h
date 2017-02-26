
void draw_omega(arma::mat & omega, arma::mat & psi,
                const arma::mat & A, const arma::vec & kappa, const arma::vec & tau,
                const arma::Col<int> & G);

void draw_S(const arma::Mat<int> & i_zeros, const arma::mat & A, const arma::Col<int> & G, 
            arma::Mat<int> & S, const arma::mat & psi, arma::vec & sumAS, const arma::vec & SCrd);

void draw_kappa_tau(const arma::mat & A, const arma::Col<int> & G, const arma::mat & omega, const arma::vec & p_kappa,
                    const arma::vec & p_tau, const arma::vec & sumAS, arma::vec & kappa, arma::vec & tau, const arma::Mat<int> & S);

double update_SuffStat_SC(const arma::Mat<int> & S, const arma::vec & kappa, const arma::vec & tau, const arma::mat & omega, 
                   int n_samples, arma::mat & exp_S, arma::vec & exp_kappa, arma::vec & exp_tau,
                   arma::vec & exp_kappa_sq, arma::vec & exp_tau_sq, arma::mat & coeff_A, arma::mat & coeff_A_sq,
                   const arma::Col<int> & G);

double gibbs_SC(int L, int N, int K, const arma::mat & A, const arma::Col<int> & G, 
              const arma::vec & p_kappa, const arma::vec & p_tau, 
              const arma::Mat<int> & i_zeros,
              arma::mat & exp_S, arma::vec & exp_kappa, arma::vec & exp_tau,
              arma::vec & exp_kappa_sq, arma::vec & exp_tau_sq,
              arma::mat & coeff_A, arma::mat & coeff_A_sq,
              const arma::vec & SCrd,
              int burn_in, int n_samples, int thin);



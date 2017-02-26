void draw_Z(const arma::Mat<int> & bulkExpr,
            const arma::mat & A,
            const arma::mat & W,
            arma::Mat<int> & Zik,
            arma::Mat<int> & Zjk);


void draw_W(const arma::vec & alpha,
            const arma::Mat<int> & Zjk,
            arma::mat & W);


void update_SuffStat_BK(const arma::Mat<int> & Zik,
                        const arma::Mat<int> & Zjk,
                        const arma::mat & W,
                        arma::mat & exp_Zik,
                        arma::mat & exp_Zjk,
                        arma::mat & exp_W,
                        arma::mat & exp_logW,
                        int n_samples);


void gibbs_BK(int K, int M, int N, const arma::Mat<int> & i_markers,
              const arma::Mat<int> & bulkExpr, const arma::vec & alpha,
              const arma::mat & A, arma::mat & exp_Zik, arma::mat & exp_Zjk,
              arma::mat & exp_W, arma::mat & exp_logW,
              int burn_in, int n_samples, int thin);

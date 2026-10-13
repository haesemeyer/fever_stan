data{
    // observed data
    int<lower=1> T;  // number of timepoints in the data
    array[T] real y;  // observed turn angles
}

parameters{
    real<lower=5, upper=100> turn_mu;  // the average of the turn gamma distribution
    real<lower=.1, upper=10> straight_std;  // the standard deviation of the straight distribution
    real<lower=.01, upper=10000> turn_var;  // the variance of the turn gamma distribution
    simplex[3] theta_mix;
}

transformed parameters{
    real turn_beta = turn_mu / turn_var;
    real turn_alpha = turn_mu * turn_mu / turn_var;
}

model{
    vector[3] lp_theta = log(theta_mix);
    vector[3] lp_evidence;
    for(t in 1:T)
    {
        lp_evidence = lp_theta; // initialize with mixing proportions
        // negative turn
        if (y[t] < 0)
            lp_evidence[1] += gamma_lpdf(-y[t] | turn_alpha, turn_beta);
        else
            lp_evidence[1] = log(0);
        // straight
        lp_evidence[2] += normal_lpdf(y[t] | 0, straight_std);
        // positive turn
        if (y[t] > 0)
            lp_evidence[3] += gamma_lpdf(y[t] | turn_alpha, turn_beta);
        else
            lp_evidence[3] = log(0);

        target += log_sum_exp(lp_evidence);
    }
}

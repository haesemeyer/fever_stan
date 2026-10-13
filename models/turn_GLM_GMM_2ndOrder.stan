data{
    // observed data
    int<lower=1> T;  // number of timepoints in the data
    array[T] real y;  // observed turn angles
    array[T] real temperatures;  // bout temperatures
    array[T] real d_temperatures;  // previous bout delta-temperatures
    array[T] real prev_angles;  // previous bout turn angle
    // GMM definition
    real turn_mu;  // the average of the turn distribution
    real straight_std;  // the standard deviation of the straight distribution
    real turn_var;  // the variance of the turn distribution
}

transformed data{
    // predictor matrix with first and second order interaction terms
    matrix[T, 10] glm_x;
    for (t in 1:T)
    {
        glm_x[t, 1] = temperatures[t];
        glm_x[t, 2] = d_temperatures[t];
        glm_x[t, 3] = temperatures[t] * d_temperatures[t];
        glm_x[t, 4] = temperatures[t] * temperatures[t];
        glm_x[t, 5] = d_temperatures[t] * d_temperatures[t];
        glm_x[t, 6] = prev_angles[t];
        glm_x[t, 7] = prev_angles[t] * temperatures[t];
        glm_x[t, 8] = prev_angles[t] * d_temperatures[t];
        glm_x[t, 9] = prev_angles[t] * temperatures[t] * d_temperatures[t];
        glm_x[t, 10] = prev_angles[t] * prev_angles[t];
    }
    // Orthogonalize our predictor matrix using QR decomposition
    matrix[T, 10] glm_X_Q_star;
    matrix[10, 10] glm_X_R_star;
    matrix[10, 10] glm_X_R_star_inv;
    glm_X_Q_star = qr_thin_Q(glm_x) * sqrt(T - 1);
    glm_X_R_star = qr_thin_R(glm_x) / sqrt(T - 1);
    glm_X_R_star_inv = inverse(glm_X_R_star);// to be able to recast our parameters in glm_X space later

    // GMM parameters for turn gamma distributions
    real turn_beta = turn_mu / turn_var;
    real turn_alpha = turn_mu * turn_mu / turn_var;
}

parameters{
    // glm parameters of GMM on glm_X_Q_star
    array[2] vector[10] gmm_mix_beta;  // coefficients for left and right
    real gmm_mix_icept;  // the shared intercept
}

model{

    ////////////////////////////////////////////
    /////////////// GMM outputs ////////////////
    ////////////////////////////////////////////
    // priors on parameters - decay towards 0
    for (k in 1:2)
    {
        gmm_mix_beta[k] ~ normal(0, 2);
    }
    gmm_mix_icept ~ normal(0, 2);
    // compute our per-step mixing coefficients
    array[T] vector[3] lp_theta;  // log mixing coefficients - 3 components
    for (t in 1:T)
    {
        array[3] real lp_mix;//re-used intermediate variable to compute one set of mixing coefs
        real lse; // place-holder for log-sum to allow soft-max calculation

        lp_mix[1] = dot_product(gmm_mix_beta[1], glm_X_Q_star[t]) + gmm_mix_icept;
        lp_mix[2] = 0;  // straight pinned
        lp_mix[3] = dot_product(gmm_mix_beta[2], glm_X_Q_star[t]) + gmm_mix_icept;
        lse = log_sum_exp(lp_mix);
        for (ek in 1:3)
        {
            lp_theta[t][ek] = lp_mix[ek] - lse;
        }
    }  // for (t in 1:T)

    ////////////////////////////////////////////
    ////////////////// MODEL ///////////////////
    ////////////////////////////////////////////

    vector[3] lp_evidence;  // For each component of the GMM holds the likelihood of the observation given that component

    for (t in 1:T)
    {
        lp_evidence = lp_theta[t];
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

generated quantities {
    // compute coefficients in original space
    array[2] vector[10] gmm_mix_orig;
    for (k in 1:2)
    {
        gmm_mix_orig[k] = glm_X_R_star_inv * gmm_mix_beta[k];
    }
}

data{
    // observed data
    int<lower=1> T;  // number of timepoints in the data
    array[T] real y;  // observed displacement
    array[T] real temperatures;  // bout temperatures
    array[T] real d_temperatures;  // previous bout delta-temperatures
    // gamma parameter
    real beta;
}

transformed data{
    // predictor matrix with first order interaction terms
    matrix[T, 5] glm_x;
    for (t in 1:T)
    {
        glm_x[t, 1] = temperatures[t];
        glm_x[t, 2] = d_temperatures[t];
        glm_x[t, 3] = temperatures[t] .* d_temperatures[t];
        glm_x[t, 4] = temperatures[t] .* temperatures[t];
        glm_x[t, 5] = d_temperatures[t] .* d_temperatures[t];
    }
    // Orthogonalize our predictor matrix using QR decomposition
    matrix[T, 5] glm_X_Q_star;
    matrix[5, 5] glm_X_R_star;
    matrix[5, 5] glm_X_R_star_inv;
    glm_X_Q_star = qr_thin_Q(glm_x) * sqrt(T - 1);
    glm_X_R_star = qr_thin_R(glm_x) / sqrt(T - 1);
    glm_X_R_star_inv = inverse(glm_X_R_star);// to be able to recast our parameters in glm_X space later
}

parameters{
    // glm parameters on glm_X_Q_star
    vector[5] glm_mu_xq;  // coefficients for calculating the gamma mean
    real glm_mu_icept;  // the corresponding intercept
}

transformed parameters{
    // compute our per-step gamma coefficients
    vector[T] mu;
    real glm_mu;
    for (t in 1:T)
    {
        //mu[t] = exp(dot_product(glm_X_Q_star[t], glm_mu_xq) + glm_mu_icept);
        glm_mu = dot_product(glm_X_Q_star[t], glm_mu_xq) + glm_mu_icept;
        mu[t] = 3000/(1+exp(-glm_mu));
    }
}

model{

    // priors on gamma parameters
    mu ~ normal(1000, 1000);

    // priors on parameters - decay towards 0
    glm_mu_xq ~ normal(0, 1);
    glm_mu_icept ~ normal(0, 1);

    for (t in 1:T)
    {
        target += gamma_lpdf(y[t] | mu[t] * beta, beta);
    }
}

generated quantities {
   // glm coefficients on input data
    vector[5] glm_mu_orig;
    glm_mu_orig = glm_X_R_star_inv * glm_mu_xq;
}

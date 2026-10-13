functions{
    int index_count(int order)
    {
        if (order == 1)
            return 2;
        else if (order == 2)
            return 5;
        else
            return 9;
    }
}

data{
    // observed data
    int<lower=1> T;  // number of timepoints in the data
    int<lower=1> M;  // number of trajectories
    array[M] int N_POINTS;  // number of datapoints in each trajectory
    array[T] int<lower=-1, upper=1> modes;  // annotated swim modes
    array[T] real temperatures;  // bout temperatures
    array[T] real d_temperatures;  // previous bout delta-temperatures
    // hyper-parameter
    int<lower=1, upper=3> glm_order;
}

transformed data{
    int glm_index_count = index_count(glm_order);

    // predictor matrix with first and second order interaction terms
    matrix[T, 9] glm_x;
    for (t in 1:T)
    {
        glm_x[t, 1] = temperatures[t];
        glm_x[t, 2] = d_temperatures[t];
        glm_x[t, 3] = temperatures[t] * d_temperatures[t];
        glm_x[t, 4] = temperatures[t] * temperatures[t];
        glm_x[t, 5] = d_temperatures[t] * d_temperatures[t];
        glm_x[t, 6] = temperatures[t] * temperatures[t] * d_temperatures[t];
        glm_x[t, 7] = temperatures[t] * d_temperatures[t] * d_temperatures[t];
        glm_x[t, 8] = temperatures[t] * temperatures[t] * temperatures[t];
        glm_x[t, 9] = d_temperatures[t] * d_temperatures[t] * d_temperatures[t];
    }
    // Orthogonalize our predictor matrix using QR decomposition
    matrix[T, glm_index_count] glm_X_Q_star;
    matrix[glm_index_count, glm_index_count] glm_X_R_star;
    matrix[glm_index_count, glm_index_count] glm_X_R_star_inv;
    glm_X_Q_star = qr_thin_Q(glm_x[:, 1:glm_index_count]) * sqrt(T - 1);
    glm_X_R_star = qr_thin_R(glm_x[:, 1:glm_index_count]) / sqrt(T - 1);
    glm_X_R_star_inv = inverse(glm_X_R_star);// to be able to recast our parameters in glm_X space later

    // cumulative sum of trajectory datapoints
    array[M+1] int NPC;
    NPC[1] = 0;
    for(i in 1:M)
    	NPC[i+1] = NPC[i] + N_POINTS[i];
}

parameters{
    // glm parameters for mode transitions on glm_X_Q_star independent transitions for reversal mode (-1)
    // general mode (0) and persistent mode (1). We specify:
    // -1 -> -1; -1 -> 0; -1 -> 1 [pinned]
    // 0 -> -1; 0 -> 0; 0 -> 1 [pinned]
    // 1 -> -1; 1 -> 0; 1 -> 1 [pinned]
    array[3] matrix[glm_index_count, 2] mm_beta; // transition glm coefficients - one set for each mode going to self and 2 other modes with last transition pinned
    array[3] vector[2] mm_icept;  // the intercepts, one set for each mode
}

model{
    // Mode order:
    // Reversal (-1)
    // General (0)
    // Persistent (1)

    ////////////////////////////////////////////
    //////// TRANSITION MATRIX OF MODEL ////////
    ////////////////////////////////////////////

    // prior on our GLM coefficients - mildly decay weights to 0 through normal prior with large standard deviation
    for (k in 1:3)
        for (i in 1:2)
            target += normal_lpdf(mm_beta[k][:, i] | 0, 10);
    for (k in 1:3)
        target += normal_lpdf(mm_icept[k] | 0, 10);
    // DEVELOP THE LOG-TRANSITION PROBABILITY MATRIX FOR EACH TIMESTEP THETA
    array[T] matrix[3, 3] lp_transit;
    for (t in 1:T)
    {
        vector[3] lp; // re-used intermediate variable to compute one set of transition probabilities
        real lse; // place-holder for log-sum to allow soft-max calculation

        // Transitions from Reversal
        lp[1] = dot_product(mm_beta[1][:, 1], glm_X_Q_star[t]) + mm_icept[1][1]; //-1 -> -1
        lp[2] = dot_product(mm_beta[1][:, 2], glm_X_Q_star[t]) + mm_icept[1][2]; //-1 -> 0
        lp[3] = 0; //Pinned -1 -> 1
        lse = log_sum_exp(lp);
        for (k in 1:3)
        {
            lp_transit[t][1, k] = lp[k] - lse;
        }

        // Transitions from General
        lp[1] = dot_product(mm_beta[2][:, 1], glm_X_Q_star[t]) + mm_icept[2][1];  //0 -> -1
        lp[2] = dot_product(mm_beta[2][:, 2], glm_X_Q_star[t]) + mm_icept[2][2];  //0 -> 0
        lp[3] = 0;  //Pinned 0 -> 1
        lse = log_sum_exp(lp);
        for (k in 1:3)
        {
            lp_transit[t][2, k] = lp[k] - lse;
        }

        // Transitions from Persistent
        lp[1] = dot_product(mm_beta[3][:, 1], glm_X_Q_star[t]) + mm_icept[3][1];  //1 -> -1
        lp[2] = dot_product(mm_beta[3][:, 2], glm_X_Q_star[t]) + mm_icept[3][2];  //1 -> 0
        lp[3] = 0;  //Pinned 1 -> 1
        lse = log_sum_exp(lp);
        for (k in 1:3)
        {
            lp_transit[t][3, k] = lp[k] - lse;
        }

    } // for (t in 1:T)

    ////////////////////////////////////////////
    ////////////////// MODEL ///////////////////
    ////////////////////////////////////////////

    // current trajectory counter
    int cur_exp = 1;

    // accumulator to sum probabilities across source-states
    array[3] real acc;
    // log marginal probabilities of the observations y up to time t and the latent state
    // being equal to k at time t; previous latent states are marginalized out (STAN doc)
    //array[T, 3] real zeta;

    for (t in 2:T)
    {
        // check if this is the start of a new trajectory
        if (t == NPC[cur_exp+1] + 1)
        {
        	// At trajectory starts our transition matrix has no influence
		    cur_exp=cur_exp+1;
        }
        else //this is a continuation of a trajectory
        {
            // Add the probability of having transitioned from the prior state at t-1 to the current state at t
            // given the transition matrix at timepoint t
            target += lp_transit[t][modes[t-1]+2, modes[t]+2];// since mode=-1 corresponds to first element
        }
    }

}

generated quantities {
    // compute coefficients in original space

    array[3] matrix[glm_index_count, 2] mm_orig;
    for (k in 1:3)
    {
        for (i in 1:2)
            mm_orig[k][:,i] = glm_X_R_star_inv * mm_beta[k][:,i];
    }
}

data{
    // observed data
    int<lower=1> T;  // number of timepoints in the data
    array[T] real y;  // observed data
}

parameters{
    real<lower=0> alpha;
    real<lower=0> beta;
}

model{
    for (t in 1:T){
        target += gamma_lpdf(y[t] | alpha, beta);
    }
}
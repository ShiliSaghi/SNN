import numpy as np

# parameters initialization---------
# Parameters------
L = 80  # number of slots
Lb = 2  # Bandwidth factor, in paper Lb ∈ [1,6] ???
Tc = 1.0  # chip time, symbol duration ???
T = 2 * Lb * Tc  # Time slot duration: 2*1
beta = 0.25  # roll-off factor-(rrc) ???
sps = 100  # samples per symbol  ???
t = np.linspace(
    0, L * T, L * 2 * Lb * sps
)  # Time vector: MaxTime: L * T = 80 * 2 = 160, No of samples: L * 2 * Lb * sps = 80*2*1*100=16000
x = np.random.binomial(n=1, p=0.5, size=L)  # np.random.randint(0, 2, L), x ∈ {0,1}
v = np.random.binomial(n=1, p=0.5)  # Target presence: Bernoulli(0.5)
Nc = 5  # number of clutters
sigma_0 = 1.0  # Power of the target component ???
tau_0 = 0  # target delay
tau_c = np.random.uniform(
    0, 4 * Tc, Nc
)  # clutter delays uniformly distributed between 0 and 4Tc
k_c = 2  # Shape parameter for Weibull distribution, κ ∈ [0.25,2]
lambda_c = 1.0  # Scale parameter for Weibull distribution, λ ∈ (0,∞)  ???

# total input data size based on papre: 60000(train) + 10000(test) = 70000

import numpy as np

B = 400e6
# parameters initialization---------
L = 80  # number of slots
Lb = 4  # Bandwidth factor, in paper Lb ∈ [1,6]
Tc = 1 / B  # chip time
T = 2 * Lb * Tc  # Time slot duration: 2 * 4 / 400e6 = 0.02e-6
beta = 0.25  # roll-off factor-(rrc)
t = np.linspace(
    0, L * T, L * 2 * Lb
)  # Time vector: MaxTime: L * T = 80 * 0.02e-6 = 1.6e-6, No of samples: L * 2 * Lb = 80*2*4=640
x = np.random.binomial(n=1, p=0.5, size=L)  # np.random.randint(0, 2, L), x ∈ {0,1}
v = np.random.binomial(n=1, p=0.5)  # Target presence: Bernoulli(0.5)
upsample = 8

channel_duration = 1e-7
channel_samples = int(channel_duration // (Tc / (upsample * 2 * Lb)))

Nc = 5  # number of clutters
sigma_0 = 2.0  # Variance (or power) of the target amplitude β0 ???
tau_0 = np.random.uniform(0, 4 * Tc, 1)  # target delay
tau_c = np.random.uniform(
    0, 4 * Tc, Nc
)  # np.random.choice(np.arange(0, int(4 * Tc)+1), size=Nc, replace=False) #np.random.uniform(0, 4 * Tc, Nc)  # clutter delays uniformly distributed between 0 and 4Tc

k_c = 2  # Shape parameter for Weibull distribution, κ ∈ [0.25,2]
lambda_c = 1.0  # Scale parameter for Weibull distribution, λ ∈ (0,∞)  ???

# total input data size based on paper: 60000(train) + 10000(test) = 70000

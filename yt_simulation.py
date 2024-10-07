import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import config


def rrc_pulse(t, beta, Tc):
    phi_t = np.zeros_like(t)

    for i in range(len(t)):
        if t[i] == 0:
            phi_t[i] = (1 - beta + 4 * beta / np.pi) / Tc
        elif abs(t[i]) == Tc / (4 * beta):
            phi_t[i] = (beta / (np.sqrt(2) * Tc)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            numerator = np.sin(np.pi * t[i] * (1 - beta) / Tc) + 4 * beta * t[
                i
            ] / Tc * np.cos(np.pi * t[i] * (1 + beta) / Tc)
            denominator = np.pi * t[i] / Tc * (1 - (4 * beta * t[i] / Tc) ** 2)
            phi_t[i] = numerator / denominator

    return phi_t


def generate_s_t(t, T, L, x, Lb, Tc, beta):

    # s_t = np.zeros_like(t)
    s_i = np.zeros((2 * Lb * len(t)))

    # Loop in each time slot
    for l in range(len(x)):
        if x[l]:
            s_i[l * 2 * Lb + Lb] = 1.0
        else:
            s_i[l * 2 * Lb] = 1.0

        # shift_time = ((l - 1) * T) + x[l - 1] * Lb * Tc
        # pulse_time = t - shift_time
        # phi_t = rrc_pulse(pulse_time, beta, Tc)
        # if l == 1:
        #     plt.figure(1, figsize=(10, 6))
        #     plt.subplot(2, 1, 1)
        #     plt.plot(pulse_time, phi_t)
        #     plt.title("φ(t)")
        #     plt.ylabel("Amp")
        #     plt.grid(True)

        # # Summation of φ(t - ...)
        # s_t += phi_t

    upsample = 8
    s_t_prime = np.zeros((len(s_i) * upsample))
    s_t_prime[::upsample] = s_i

    pulse_length = 62
    time = np.arange(-pulse_length + 1, pulse_length) * T / (2 * Lb * upsample)
    phi_t = rrc_pulse(time, beta, T / (2 * Lb))

    # plt.plot(phi_t)
    # plt.show()

    s_t = np.convolve(s_t_prime, phi_t, "same")

    # plt.stem(np.arange(len(s_i)), s_i)
    # plt.plot(np.arange(len(s_i) * upsample) / upsample, s_t, "r")
    # plt.show()

    return t, s_t


def channel_response(t, beta, Tc, v, sigma_0, tau_0, tau_c, k_c, lambda_c, N_c):

    h_t = np.zeros_like(t, dtype=complex)

    beta_0 = np.sqrt(sigma_0 / 2) * (
        np.random.randn() + 1j * np.random.randn()
    )  # β0 ∼ CN(0,σ2)

    beta_c = np.sqrt(lambda_c * (np.random.weibull(k_c, N_c)) / 2) * np.exp(
        1j * 2 * np.pi * np.random.rand(N_c)
    )  # the amp of Nc clutters which are independent with uniform phases and Weibull absolute values
    # the amplitudes {βc }Nc are all i.i.d. βc ∼ CN(0,1) variable ???

    if v == 1:
        shift_time = t - tau_0
        g_t = rrc_pulse(-shift_time, beta, Tc)
        h_t = beta_0 * g_t

    for c in range(N_c):
        shift_time_c = t - tau_c[c]
        g_t_c = rrc_pulse(-shift_time_c, beta, Tc)
        h_t += beta_c[c] * g_t_c

    return h_t


def received_signal(s_t, h_t):
    # Convolve s(t) with h(t)
    # y = convolve(s_t, h_t, mode='full')
    h_real = h_t.real
    h_imag = h_t.imag
    y_real = np.convolve(s_t, h_real, mode="full")
    y_imag = np.convolve(s_t, h_imag, mode="full")
    y = y_real + 1j * y_imag

    return y


# Create signals----------
h_t = channel_response(
    config.t,
    config.beta,
    config.Tc,
    config.v,
    config.sigma_0,
    config.tau_0,
    config.tau_c,
    config.k_c,
    config.lambda_c,
    config.Nc,
)
print("length of h(t):", len(h_t))

plt.plot(np.abs(h_t))
plt.show()

t, s_t = generate_s_t(
    config.t, config.T, config.L, config.x, config.Lb, config.Tc, config.beta
)
print("length of s(t):", len(s_t))

y_t = received_signal(s_t, h_t)
print("length of y(t):", len(y_t))
time_y = np.arange(len(y_t))


# Add noise to y(t)--------
N0 = 0.1  # power spectral density
B = 1 / config.Tc  # Pulse bandwidth
num_noise_samples = len(y_t)
z = np.random.normal(0, np.sqrt(N0 * B), num_noise_samples) + 1j * np.random.normal(
    0, np.sqrt(N0 * B), num_noise_samples
)
y_noisy = y_t + z


# Sampling y(t) to obtain yi based on Tc----
num_samples_to_take = 2 * config.Lb * config.L  # total number of samples
sample_indexes = (
    np.arange(1, num_samples_to_take + 1) * config.Tc
)  # array of sampling indexes with Tc intervals
yi_indexes = (sample_indexes / (config.T / config.sps)).astype(
    int
)  # convert sample indexes to the corresponding indexes in y_t
yi = y_noisy[yi_indexes]


# Create yl, a list vector to hold samples for each time slot(l)----
yl = []
for l in range(1, config.L + 1):
    start_index = (l - 1) * 2 * config.Lb
    end_index = 2 * l * config.Lb
    samples_for_slot = yi[start_index:end_index]
    yl.append(samples_for_slot)


print("Length of yi:", len(yi))
print("Length of yl (number of slots):", len(yl))  # should be L=80
print(
    "Samples for first slot (y1):", yl[0]
)  # just for test and checking the first slot


# Plot s(t)---------------
plt.figure(1, figsize=(10, 6))
plt.subplot(2, 1, 2)
plt.plot(t, s_t)
bar_width = config.T / 4
x_positions = np.arange(1, len(config.x) + 1) * config.T  # Position for each slot
plt.bar(
    x_positions - bar_width / 2,
    config.x,
    width=bar_width,
    color="red",
    alpha=0.8,
    label="Bit Vector x",
)  # bar chart for bit vector x

plt.title("s(t) with bit vector x")
plt.title("S(t)")
plt.xlabel("Time")
plt.ylabel("Amp")
plt.legend()
plt.grid(True)


# Plot h(t)---------------
# --------------real------
plt.figure(2, figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(config.t, h_t.real)
plt.title("h(t)_Real")
plt.ylabel("Amplitude")
plt.grid(True)
# --------imaginary-------
plt.subplot(2, 1, 2)
plt.plot(config.t, h_t.imag)
plt.title("h(t)_Imaginary")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot y(t)---------------
# --------------real------
plt.figure(3, figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time_y, y_t.real)
plt.title("y(t)_Real")
plt.ylabel("Amplitude")
plt.grid(True)
# -------------imaginary--
plt.subplot(2, 1, 2)
plt.plot(time_y, y_t.imag)
plt.title("y(t)_Imaginary")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)


# Plot yi-------------------
# --------------real------
plt.figure(4, figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.stem(np.arange(len(yi)), yi.real, basefmt=" ", use_line_collection=True)
plt.title("yi_Real")
plt.ylabel("Value")
plt.grid(True)
# -------------imaginary--
plt.subplot(2, 1, 2)
plt.stem(np.arange(len(yi)), yi.imag, basefmt=" ", use_line_collection=True)
plt.title("yi_Imaginary")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)


plt.tight_layout()
plt.show()

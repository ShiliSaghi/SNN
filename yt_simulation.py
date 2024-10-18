import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import config


# np.random.seed(42) #For having fixed values every time the code is run.


def generate_complex_gaussian(n, mean, variance):
    real_part = np.random.normal(mean, np.sqrt(variance / 2), n)
    imag_part = np.random.normal(mean, np.sqrt(variance / 2), n)

    return real_part + 1j * imag_part


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

    # s_i = np.zeros((2 * Lb * len(t)))
    s_i = np.zeros(len(t), dtype=complex)

    # Loop in each time slot
    for l in range(len(x)):
        if x[l]:
            s_i[l * 2 * Lb + Lb] = 1.0
        else:
            s_i[l * 2 * Lb] = 1.0

    s_t_prime = np.zeros((len(s_i) * config.upsample))
    s_t_prime[:: config.upsample] = s_i

    pulse_length = 101
    time = np.arange(pulse_length) - (pulse_length - 1) // 2
    Ts = config.upsample
    phi_t = (
        np.sinc(time / Ts)
        * np.cos(np.pi * beta * time / Ts)
        / (1 - (2 * beta * time / Ts) ** 2 + 1e-8)
    ).astype(complex)

    plt.figure(1)
    plt.stem(time, phi_t)
    plt.grid()
    plt.show()

    s_t = np.convolve(s_t_prime, phi_t, "same")

    plt.figure()
    plt.title("s(i)/s(t)")
    plt.stem(Tc * np.arange(len(s_i)), s_i, label="s(i)", basefmt="b-")
    plt.plot(
        (Tc / config.upsample) * np.arange(len(s_i) * config.upsample),
        s_t,
        "r",
        label="s(t)",
    )
    plt.legend(loc="upper right")
    plt.show()

    return t, s_t


def channel_response(beta, Tc, v, sigma_0, tau_0, tau_c, Nc):

    beta_0 = generate_complex_gaussian(1, mean=0, variance=1)
    beta_c = generate_complex_gaussian(Nc, mean=0, variance=sigma_0)
    print("B0: ", beta_0)
    print("Bc: ", beta_c)

    # pulse_length = 101
    # time = np.arange(pulse_length) - (pulse_length - 1) // 2
    # Ts = config.upsample
    # phi_rx = (
    #     np.sinc(time / Ts)
    #     * np.cos(np.pi * beta * time / Ts)
    #     / (1 - (2 * beta * time / Ts) ** 2 + 1e-8)
    # )

    h_t = np.zeros(config.channel_samples)

    upsampled_time = np.arange(len(h_t)) * (Tc / (2 * config.Lb * config.upsample))

    # generate the delays
    tau_0 = config.tau_0
    tau_c = config.tau_c
    tau_0_dis = np.digitize(tau_0, upsampled_time)
    tau_c_dis = np.digitize(tau_c, upsampled_time)

    h_t[tau_0_dis] = beta_0
    h_t[tau_c_dis] = beta_c

    # h_t = np.convolve(h_t, phi_rx, "same")

    plt.plot((Tc / config.upsample) * np.arange(len(h_t)), np.abs(h_t))
    plt.show()

    # pulse_length = 62
    # time = (
    #     np.arange(-pulse_length + 1, pulse_length)
    #     * config.T
    #     / (2 * config.Lb * config.upsample)
    # )
    # shift_time = time - tau_0  # tau_0 = 0
    # g_t_0 = rrc_pulse(-shift_time, beta, Tc)  # Tc = T/2lb

    # v = 1
    # h_t_target = np.zeros(len(g_t_0))
    # if v == 1:
    #     h_t_target = beta_0 * g_t_0
    #     print("len(h_t_target)", len(h_t_target))

    # plt.figure()
    # plt.title("h(t)_target")
    # plt.plot(h_t_target, "g")
    # plt.show()

    # # for c in range(N_c):
    # #     shift_time_c = t - tau_c[c]
    # #     g_t_c = rrc_pulse(-shift_time_c, beta, Tc)
    # #     h_t += beta_c[c] * g_t_c

    # # new code-----------------------
    # g_t_c = rrc_pulse(-time, beta, Tc)

    # ls = np.linspace(0, 4 * config.Tc, 2 * config.Lb * Nc)  # 20
    # bc_sig = np.zeros(len(ls), dtype=complex)
    # # print("ls: ", ls)

    # matched_ls_index = []
    # matched_tau_index = []
    # used_ls_index = []

    # for i, tau in enumerate(
    #     tau_c
    # ):  # find the the closest index value in ls which matches tau

    #     sorted_index = np.argsort(
    #         np.abs(ls - tau)
    #     )  # sort indexes of min differences (in ascending order)

    #     for min_index in sorted_index:

    #         if min_index not in used_ls_index:
    #             matched_ls_index.append(min_index)
    #             matched_tau_index.append(i)
    #             used_ls_index.append(min_index)  # mark min_index as used
    #             break

    # for ls_idx, tau_idx in zip(matched_ls_index, matched_tau_index):
    #     bc_sig[ls_idx] = beta_c[tau_idx]
    #     # tau_c[tau_idx] = -1
    #     # print(f"tau_c[{tau_idx}] = {tau_c[tau_idx]} is closest to ls[{ls_idx}] = {ls[ls_idx]}")

    # bc_sig_prime = np.zeros((len(bc_sig) * config.upsample), dtype=complex)  # 160
    # bc_sig_prime[:: config.upsample] = bc_sig

    # plt.figure()
    # plt.title("bc_sig_prime")
    # plt.stem(bc_sig_prime, basefmt="b-")
    # # plt.show()

    # h_t_clutter = np.convolve(bc_sig_prime, g_t_c, "same")
    # print("len(h_t_clutter): ", len(h_t_clutter))

    # plt.figure()
    # plt.title("h(t)_clutter")
    # plt.stem(np.arange(len(bc_sig)), bc_sig, label="bc_sig", basefmt="b-")
    # plt.plot(
    #     np.arange(len(bc_sig) * config.upsample) / config.upsample,
    #     h_t_clutter,
    #     "r",
    #     label="h(t)_clutter",
    # )
    # plt.legend()
    # # plt.show()

    # # Match the lengths of both parts of the h(t) signal
    # max_len = max(len(h_t_target), len(h_t_clutter))
    # h_t_target_padded = np.pad(h_t_target, (0, max_len - len(h_t_target)), "constant")
    # h_t_clutter_padded = np.pad(
    #     h_t_clutter, (0, max_len - len(h_t_clutter)), "constant"
    # )

    # h_t = h_t_target_padded + h_t_clutter_padded
    # print("len(h_t): ", len(h_t))

    # plt.figure()
    # plt.title("h(t)")
    # plt.plot(h_t_target_padded, "g", label="h(t)_target")
    # plt.plot(h_t_clutter, "r", label="h(t)_clutter")
    # plt.plot(h_t, "b", label="h(t)")
    # plt.legend()
    # plt.show()

    return h_t


def received_signal(s_t, h_t):

    y = convolve(s_t, h_t, mode="full")

    # h_real = h_t.real
    # h_imag = h_t.imag
    # y_real = np.convolve(s_t, h_real, mode="full")
    # y_imag = np.convolve(s_t, h_imag, mode="full")
    # y = y_real + 1j * y_imag

    return y


def sample_received_signal(y, h):
    first_nonzero = np.argwhere(np.abs(h))[0][0]
    y = y[first_nonzero :: config.upsample]
    plt.plot(np.abs(h))
    plt.title(first_nonzero)
    plt.show()
    return y


# --main--
# Create signals----------
t, s_t = generate_s_t(
    config.t, config.T, config.L, config.x, config.Lb, config.Tc, config.beta
)


h_t = channel_response(
    config.beta,
    config.Tc,
    config.v,
    config.sigma_0,
    config.tau_0,
    config.tau_c,
    config.Nc,
)


y_t = received_signal(s_t, h_t)
print("length of y(t):", len(y_t))
time_y = np.arange(len(y_t))

plt.plot((config.Tc / config.upsample) * np.arange(len(s_t)), s_t.real)
plt.plot((config.Tc / config.upsample) * np.arange(len(h_t)), np.abs(h_t))
plt.plot((config.Tc / config.upsample) * np.arange(len(y_t)), y_t.real)
plt.show()

# Plot y(t)---------------
# --------------real------
plt.figure(7, figsize=(10, 6))
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
plt.show()


# Add noise to y(t)--------
N0 = 0.1  # power spectral density
B = 1 / config.Tc  # Pulse bandwidth
num_noise_samples = len(y_t)

z = np.random.normal(0, np.sqrt(N0 * B), num_noise_samples) + 1j * np.random.normal(
    0, np.sqrt(N0 * B), num_noise_samples
)
y_noisy = y_t + z

y_sampled = sample_received_signal(y_noisy, h_t)

# # Sampling y(t) to obtain yi based on Tc----
# num_samples_to_take = 2 * config.Lb * config.L  # total number of samples
# sample_indexes = (
#     np.arange(1, num_samples_to_take + 1) * config.Tc
# )  # array of sampling indexes with Tc intervals
# yi_indexes = (sample_indexes / (config.T / config.sps)).astype(
#     int
# )  # convert sample indexes to the corresponding indexes in y_t
# # yi = y_noisy[yi_indexes]


# Create yl, a list vector to hold samples for each time slot(l)----
# yl = []
# for l in range(1, config.L + 1):
#     start_index = (l - 1) * 2 * config.Lb
#     end_index = 2 * l * config.Lb
#     samples_for_slot = yi[start_index:end_index]
#     yl.append(samples_for_slot)


# print("Length of yi:", len(yi))
# print("Length of yl (number of slots):", len(yl))  # should be L=80
# print(
#     "Samples for first slot (y1):", yl[0]
# )  # just for test and checking the first slot


# # Plot s(t)---------------
# plt.figure(1, figsize=(10, 6))
# plt.subplot(2, 1, 2)
# plt.plot(t, s_t)
# bar_width = config.T / 4
# x_positions = np.arange(1, len(config.x) + 1) * config.T  # Position for each slot
# plt.bar(
#     x_positions - bar_width / 2,
#     config.x,
#     width=bar_width,
#     color="red",
#     alpha=0.8,
#     label="Bit Vector x",
# )  # bar chart for bit vector x

# plt.title("s(t) with bit vector x")
# plt.title("S(t)")
# plt.xlabel("Time")
# plt.ylabel("Amp")
# plt.legend()
# plt.grid(True)


# # Plot h(t)---------------
# # --------------real------
# plt.figure(2, figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(config.t, h_t.real)
# plt.title("h(t)_Real")
# plt.ylabel("Amplitude")
# plt.grid(True)
# # --------imaginary-------
# plt.subplot(2, 1, 2)
# plt.plot(config.t, h_t.imag)
# plt.title("h(t)_Imaginary")
# plt.ylabel("Amplitude")
# plt.grid(True)


# # Plot yi-------------------
# # --------------real------
# plt.figure(4, figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.stem(np.arange(len(yi)), yi.real, basefmt=" ", use_line_collection=True)
# plt.title("yi_Real")
# plt.ylabel("Value")
# plt.grid(True)
# # -------------imaginary--
# plt.subplot(2, 1, 2)
# plt.stem(np.arange(len(yi)), yi.imag, basefmt=" ", use_line_collection=True)
# plt.title("yi_Imaginary")
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.grid(True)


# plt.tight_layout()
# plt.show()

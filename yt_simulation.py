import pickle
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

    # plt.figure(1)
    # plt.stem(time, phi_t)
    # plt.grid()
    # plt.show()

    s_t = np.convolve(s_t_prime, phi_t, "same")

    # plt.figure()
    # plt.title("s(i)/s(t)")
    # plt.stem(Tc * np.arange(len(s_i)), s_i, label="s(i)", basefmt="b-")
    # plt.plot(
    #     (Tc / config.upsample) * np.arange(len(s_i) * config.upsample),
    #     s_t,
    #     "r",
    #     label="s(t)",
    # )
    # plt.legend(loc="upper right")
    # plt.show()

    return t, s_t


def channel_response(beta, Tc, v, sigma_0, tau_0, tau_c, Nc):

    beta_0 = generate_complex_gaussian(1, mean=0, variance=1)
    beta_c = generate_complex_gaussian(Nc, mean=0, variance=sigma_0)
    # print("B0: ", beta_0)
    # print("Bc: ", beta_c)

    h_t = np.zeros(config.channel_samples, dtype = complex)

    upsampled_time = np.arange(len(h_t)) * (Tc / (config.upsample))

    # generate the delays
    tau_0 = config.tau_0 #finds the indices in upsampled_time that each value in tau_0 would fall into.
    tau_c = config.tau_c
    tau_0_dis = np.digitize(tau_0, upsampled_time)
    tau_c_dis = np.digitize(tau_c, upsampled_time)

    h_t[tau_0_dis] = v * beta_0
    h_t[tau_c_dis] = beta_c

    return h_t


def received_signal(s_t, h_t):

    y = convolve(s_t, h_t, mode="same")
    return y


def sample_received_signal(y, h, st):
    first_nonzero = np.argwhere(np.abs(h))[0][0] #return the indices of non-zero elements
    # plt.plot(
    #     np.arange(len(y[first_nonzero:])) / config.upsample,
    #     y[first_nonzero:].real,
    #     "k--",
    # )

    # Calculate if padding is needed
    if (len(y) - first_nonzero) % config.upsample != 0:
        padding_length = config.upsample - ((len(y) - first_nonzero) % config.upsample)
        y = np.pad(y, (0, padding_length), mode='constant')
        
    y_sampled = y[first_nonzero :: config.upsample]
    # plt.plot(np.abs(h))
    # plt.title(first_nonzero)
    # plt.show()
    # plt.stem(y.real)
    # plt.show()

    # print("length of y-sampled(i): ", len(y_sampled))

    # #Plot y_sampled---------------
    # plt.figure(8, figsize=(10, 6))
    # # Real part
    # plt.subplot(2, 1, 1)
    # plt.plot(y_noisy.real, label="y_noisy")
    # plt.scatter(np.arange(first_nonzero, len(y_noisy), 8), y_sampled.real, color='red', label="y_sampled")
    # plt.title("y_Real")
    # plt.legend()
    # # Imaginary part
    # plt.subplot(2, 1, 2)
    # plt.plot(y_noisy.imag, label="y_noisy Imaginary")
    # plt.scatter(np.arange(first_nonzero, len(y_noisy), 8), y_sampled.imag, color='red', label="y_sampled", s=20)
    # plt.title("y_Imaginary")
    # plt.xlabel("Time")
    # plt.legend()
    # #plt.show()

    yl = y_sampled.reshape(config.L, 2*config.Lb)
    # print("yl(i) shape: ", yl.shape)
    yl_mapped = complex_to_real_vector(yl)
    # print("yl_mapped shape: ", yl_mapped.shape)
   

    xl = st[ :: config.upsample].reshape(config.L, 2*config.Lb)
    # print("xl(i) shape: ", xl.shape)
    xl_mapped = complex_to_real_vector(xl)
    # print("xl_mapped shape: ", xl_mapped.shape)


    # plt.figure(figsize=(10, 3))
    # plt.plot(yl_mapped[:, :8].real.flatten())
    # plt.title("Real Parts of All Rows in yl_mapped")
    plt.show()

    return yl_mapped, xl_mapped


def create_noise(num_noise_samples):
    Eb = 1
    SNR = 10  # SNR is given 10 dB
    squared_norm_h = np.abs(h_t) ** 2  #|h|^2
    average_squared_norm_h = np.mean(squared_norm_h)
    N0B = (average_squared_norm_h * Eb) / SNR
    # print(f"_____N0B: {N0B}")
    noise = np.random.normal(0, np.sqrt(N0B), num_noise_samples) + 1j * np.random.normal(0, np.sqrt(N0B), num_noise_samples)
    return noise


def complex_to_real_vector(yl):
    y_mapped=[]
    for row in yl:
        real_part = row.real
        imag_part = row.imag
        row_concat = np.concatenate((real_part,imag_part))

        y_mapped.append(row_concat)
    
    return np.array(y_mapped)




#------------------main------------------
num_round = 100000
multiple_tuples=[]

for r in range(num_round):
    x = np.random.binomial(n=1, p=0.5, size=config.L)  # np.random.randint(0, 2, L), x ∈ {0,1}
    v = np.random.binomial(n=1, p=0.5)  # Target presence: Bernoulli(0.5)

    #create signals---
    t, s_t = generate_s_t(
        config.t, config.T, config.L, x, config.Lb, config.Tc, config.beta
    )
    # print("length of s(t):", len(s_t))


    h_t = channel_response(
        config.beta,
        config.Tc,
        v,
        config.sigma_0,
        config.tau_0,
        config.tau_c,
        config.Nc,
    )
    # print("length of h(t):", len(h_t))


    y_t = received_signal(s_t, h_t)
    # print("length of y(t):", len(y_t))


    # plt.plot((config.Tc / config.upsample) * np.arange(len(s_t)), s_t.real, label = "s(t)")
    # plt.plot((config.Tc / config.upsample) * np.arange(len(h_t)), np.abs(h_t), label = "h(t)")
    # plt.plot((config.Tc / config.upsample) * np.arange(len(y_t)), y_t.real, label = "y(t)")
    # plt.legend()
    # plt.show()


    # Add noise to y(t)--------
    z = create_noise(len(y_t))
    y_noisy = y_t + z


    # Downsample the y(t)------
    y, x = sample_received_signal(y_noisy, h_t, s_t)

    print("round# :", r, " v : ", v)
    multiple_tuples.append((y,x,v))

# # Print each tuple---
# for i, t in enumerate(multiple_tuples):
#     print(f"Tuple {i + 1}:")
#     print(f"y (shape {t[0].shape}):\n{t[0]}")
#     print(f"x (shape {t[1].shape}):\n{t[1]}")
#     print(f"v: {t[2]}\n")
# print(multiple_tuples)

#Create dataset---------------
with open("./dataset.pkl", "wb") as f:
    pickle.dump(multiple_tuples, f)

# Load the dataset
with open("./dataset.pkl", "rb") as f:
    loaded_tuples = pickle.load(f)

#plot dataset----
y_test, x_test, v_test = loaded_tuples[3]
print(y_test.shape, x_test.shape, v_test)
y_test_real = y_test[:, :8].real.flatten()
plt.figure(figsize=(10, 6))
plt.plot(y_test_real, label="y")
plt.stem( x_test[:, :8].flatten(), label="x", linefmt="red", markerfmt="ro", basefmt="r-")
plt.title(f"Third element of the tuple, v = {v_test}")
plt.legend()
plt.show()








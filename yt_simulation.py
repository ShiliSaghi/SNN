import csv
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

    print("length of s_i:", len(s_i))
    s_i_2D = s_i.reshape(80,8)
    print("shape s_i_2D:", s_i_2D.shape)
    s_i_reshape = complex_to_real_vector(s_i_2D)
    print("shape s_i_reshape:", s_i_reshape.shape)

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
    print("B0: ", beta_0)
    print("Bc: ", beta_c)

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


def sample_received_signal(y, h):
    first_nonzero = np.argwhere(np.abs(h))[0][0] #return the indices of non-zero elements
    # plt.plot(
    #     np.arange(len(y[first_nonzero:])) / config.upsample,
    #     y[first_nonzero:].real,
    #     "k--",
    # )

    # print("!!!!!!!!!!!!!!!!!!", (len(y) - first_nonzero) % config.upsample)
    # Calculate if padding is needed
    # if (len(y) - first_nonzero) % config.upsample != 0:
    #     padding_length = config.upsample - ((len(y) - first_nonzero) % config.upsample)
    #     y = np.pad(y, (0, padding_length), mode='constant')
        
    y = y[first_nonzero :: config.upsample]
    # Calculate if padding is needed
    if len(y) % config.upsample != 0:
        padding_length = config.upsample - len(y) % config.upsample
        y = np.pad(y, (0, padding_length), mode='constant')
        
    # plt.plot(np.abs(h))
    # plt.title(first_nonzero)
    # plt.show()
    # plt.stem(y.real)

    # plt.show()
    return y


def complex_to_real_vector(yl):
    y_mapped=[]
    for row in yl:
        real_part = row.real
        imag_part = row.imag
        row_concat = np.concatenate((real_part,imag_part)).reshape(16)

        y_mapped.append(row_concat)
    
    return np.array(y_mapped)


def generate_y(x, v):

    # Create signals---
    t, s_t = generate_s_t(
        config.t, config.T, config.L, config.x, config.Lb, config.Tc, config.beta
    )
    print("length of s(t):", len(s_t))  

    h_t = channel_response(
        config.beta,
        config.Tc,
        v,
        config.sigma_0,
        config.tau_0,
        config.tau_c,
        config.Nc,
    )
    print("length of h(t):", len(h_t))

    y_t = received_signal(s_t, h_t)
    print("length of y(t):", len(y_t))
    time_y = np.arange(len(y_t))

    # plt.plot((config.Tc / config.upsample) * np.arange(len(s_t)), s_t.real, label = "s(t)")
    # plt.plot((config.Tc / config.upsample) * np.arange(len(h_t)), np.abs(h_t), label = "h(t)")
    # plt.plot((config.Tc / config.upsample) * np.arange(len(y_t)), y_t.real, label = "y(t)")
    # plt.legend()
    # plt.show()

    # Add noise to y(t)--------
    Eb = 1
    SNR = 10  # SNR is given 10 dB
    squared_norm_h = np.abs(h_t) ** 2  #|h|^2
    average_squared_norm_h = np.mean(squared_norm_h)
    N0B = (average_squared_norm_h * Eb) / SNR
    print(f"_____N0B: {N0B}")

    num_noise_samples = len(y_t)
    z = np.random.normal(0, np.sqrt(N0B), num_noise_samples) + 1j * np.random.normal(0, np.sqrt(N0B), num_noise_samples)
    y_noisy = y_t + z
    print("length of y_noisy:", len(y_noisy))

    # #Plot y_noisy---------------
    # #real-------
    # plt.figure(8, figsize=(10, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(y_noisy.real)
    # plt.title("y_noisy_Real")
    # #imaginary---
    # plt.subplot(2, 1, 2)
    # plt.plot(y_noisy.imag)
    # plt.title("y_noisy_Imaginary")
    # plt.xlabel("Time")
    # # plt.show()


    # Sampling of y(t)-----------
    y_sampled = sample_received_signal(y_noisy, h_t)
    print("length of y-sampled(i): ", len(y_sampled))
    # #Plot y_sampled---------------
    # #real-------
    # plt.figure(8, figsize=(10, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(y_sampled.real)
    # plt.title("y_noisy_Real")
    # #imaginary---
    # plt.subplot(2, 1, 2)
    # plt.plot(y_sampled.imag)
    # plt.title("y_noisy_Imaginary")
    # plt.xlabel("Time")
    # # plt.show()

    yl = y_sampled.reshape(config.L, 2*config.Lb)
    print("yl(i) shape: ", yl.shape)
    # print(yl)
    yl_mapped = complex_to_real_vector(yl)
    print("yl_mapped shape: ", yl_mapped.shape)
    # print(yl_mapped)

    # yl_mapped_real = yl_mapped[:, :8].real.flatten()
    # plt.figure(figsize=(10, 3))
    # plt.plot(yl_mapped_real)
    # plt.title("Real Parts of All Rows in yl_mapped")
    # plt.show()

    return yl_mapped



# -----------------main------------------
num_round = 5 #100000
multiple_tuples=[]

for r in range(num_round):
    x = np.random.binomial(n=1, p=0.5, size=config.L)  # np.random.randint(0, 2, L), x ∈ {0,1}
    v = np.random.binomial(n=1, p=0.5)  # Target presence: Bernoulli(0.5)
    y = generate_y(x, v)

    print("round# :", r, "\n v : ", v)
    x_reshape= np.zeros((80,16))
    x_reshape[:, 0]= config.x
    #print("x_reshape: ", x_reshape.shape)

    multiple_tuples.append((y,x_reshape,v))

# # Print each tuple---
# for i, t in enumerate(multiple_tuples):
#     print(f"Tuple {i + 1}:")
#     print(f"y (shape {t[0].shape}):\n{t[0]}")
#     print(f"x (shape {t[1].shape}):\n{t[1]}")
#     print(f"v: {t[2]}\n")
    

#Create dataset---------------
with open("./dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for y, x, v in multiple_tuples:
        writer.writerow([y.tolist() if hasattr(y, "tolist") else y,
                         x.tolist() if hasattr(x, "tolist") else x,
                         v])

# Load from the CSV file (basic loading; adjust as needed)
with open("./dataset.csv", "r") as f:
    reader = csv.reader(f)
    loaded_tuples = [row for row in reader]

print(loaded_tuples[7:11])






import numpy as np
from .constants import *


def _calculate_P_internal(pm, T, B_calc, SUM1, K0, G0, Q0, F0, U0):
    """根据给定的摩尔密度pm计算压力P的内部辅助函数"""
    pr = (K0**3) * pm

    # 向量化计算 SUM2
    n_range = np.arange(12, 58)
    Cn_vec = a[n_range] * ((G0 + 1 - g[n_range])**g[n_range]) * \
        (((Q0**2) + 1 - q[n_range])**q[n_range]) * \
        ((F0 + 1 - f[n_range])**f[n_range]) * \
        (U0**u[n_range]) * (T**(-u[n_range]))

    pr_k = pr**k[n_range]
    term_vec = (b[n_range] - c[n_range] * k[n_range] * pr_k) * \
        (pr**b[n_range]) * np.exp(-c[n_range] * pr_k)

    SUM2 = np.sum(Cn_vec * term_vec)

    P = pm * R * T * (1 + B_calc * pm - pr * SUM1 + SUM2)
    return P, pr


def calculate_z_factor_bisection(T, P0, x, max_iterations=1000, tolerance=0.00001, log_callback=None):
    """
    使用二分法计算天然气压缩因子Z。
    """

    # Part 1: 计算第二维利系数 B
    B_calc = 0.0
    E_outer = np.sqrt(np.outer(E, E))
    G_outer = np.add.outer(G, G) / 2
    Q_outer = np.outer(Q, Q)
    F_outer_sqrt = np.sqrt(np.outer(F, F))
    S_outer = np.outer(S, S)
    W_outer = np.outer(W, W)
    K_outer_pow1_5 = np.outer(K, K)**1.5
    x_outer = np.outer(x, x)

    Eij = Ex * E_outer
    Gij = Gx * G_outer

    for n in range(18):
        ZJCS = T**(-u[n])

        Bij = ((Gij + 1 - g[n])**g[n]) * \
              ((Q_outer + 1 - q[n])**q[n]) * \
              ((F_outer_sqrt + 1 - f[n])**f[n]) * \
              ((S_outer + 1 - s[n])**s[n]) * \
              ((W_outer + 1 - w[n])**w[n])

        sum_val = np.sum(x_outer * Bij * (Eij**u[n]) * K_outer_pow1_5)
        B_calc += a[n] * ZJCS * sum_val

    # print(f"计算出的第二维利系数 B = {B_calc}") # 在GUI模式下移除打印

    # Part 2: 计算 Cn 所需的中间变量
    F0 = np.sum(x**2 * F)
    Q0 = np.sum(x * Q)
    sum1_G = np.sum(x * G)
    sum2_E = np.sum(x * E**2.5)

    G0_term = np.triu(x_outer * (Gx - 1) * np.add.outer(G, G), k=1)
    G0 = sum1_G + np.sum(G0_term)

    U0_term = np.triu(x_outer * (Ux**5 - 1) * (np.outer(E, E)**2.5), k=1)
    U0 = (sum2_E**2 + np.sum(U0_term))**0.2

    # print("中间变量计算完成: F0, Q0, G0, U0")

    # Part 3: 计算 K0
    sum1_K = np.sum(x * K**2.5)
    sum2_K_term = np.triu(x_outer * (Kx**5 - 1) * (np.outer(K, K)**2.5), k=1)
    K0 = (sum1_K**2 + 2 * np.sum(sum2_K_term))**0.2
    # print(f"K0 计算完成: K0 = {K0}")

    # Part 4 & 5: 使用二分法迭代计算压力 P
    n_range_sum1 = np.arange(12, 18)
    Cn_vec_sum1 = a[n_range_sum1] * ((G0 + 1 - g[n_range_sum1])**g[n_range_sum1]) * \
        (((Q0**2) + 1 - q[n_range_sum1])**q[n_range_sum1]) * \
        ((F0 + 1 - f[n_range_sum1])**f[n_range_sum1]) * \
        (U0**u[n_range_sum1]) * (T**(-u[n_range_sum1]))
    SUM1 = np.sum(Cn_vec_sum1)

    if log_callback:
        log_callback("开始压力迭代计算 (二分法)...\n")
    iteration_count = 0
    pm_low = 0.0
    pm_high = 100.0  # 设定一个足够大的上界
    pm = 0.0
    P = 0.0
    pr = 0.0

    P_low, _ = _calculate_P_internal(
        pm_low, T, B_calc, SUM1, K0, G0, Q0, F0, U0)
    P_high, _ = _calculate_P_internal(
        pm_high, T, B_calc, SUM1, K0, G0, Q0, F0, U0)
    if not (P_low < P0 < P_high) and log_callback:
        log_callback(
            f"警告: 目标压力 P0={P0} 不在初始搜索区间 [{P_low:.4f}, {P_high:.4f}] 内。\n")

    while iteration_count < max_iterations:
        pm = (pm_low + pm_high) / 2
        P, pr = _calculate_P_internal(pm, T, B_calc, SUM1, K0, G0, Q0, F0, U0)

        if log_callback:
            log_message = f"  迭代 {iteration_count+1}: 区间[{pm_low:.6f}, {pm_high:.6f}], 中点pm={pm:.6f}, 计算P={P:.6f}, 差值={abs(P - P0):.10f}\n"
            log_callback(log_message)

        if abs(P - P0) < tolerance:
            break

        if P < P0:
            pm_low = pm
        else:
            pm_high = pm

        iteration_count += 1

    if iteration_count == max_iterations and log_callback:
        log_callback("警告: 已达到最大迭代次数，结果可能不准确。\n")
    elif log_callback:
        log_callback(f"迭代完成，共 {iteration_count+1} 次。\n")

    # Part 6: 计算最终结果
    Z = P0 / (pm * R * T)
    M0 = np.sum(x * M)
    p_density = M0 * pm

    return Z, pm, pr, p_density, iteration_count


if __name__ == '__main__':
    # 默认参数
    T_in = 293.15
    P0_in = 0.101325
    x_in = np.array([0.961651, 0.008606, 0.004567, 0.01998, 0.003859, 0,
                     0, 0, 0, 0, 0, 0.000950, 0, 0.000138, 0.000249, 0, 0, 0, 0, 0, 0])

    calculate_z_factor_bisection(
        T_in, P0_in, x_in, tolerance=0.00001, log_callback=print)

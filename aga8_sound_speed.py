import numpy as np
import math
from typing import Dict, Tuple

# 导入常量
from aga8_calculator.constants import (
    b, k, c, g, f, q, s, w, a, u,
    M, E, G, Q, K, F, S, W,
    Ex, Gx, Ux, Kx,
    N, R
)

# ----------------- 理想气 cp0 系数 (来自 test.py/main.py) -----------------
# 注意：顺序必须与 aga8_calculator.constants 中的组分顺序一致
# 顺序: CH4, N2, CO2, C2H6, C3H8, H2O, H2S, H2, CO, O2, i-C4, n-C4, i-C5, n-C5, n-C6, n-C7, n-C8, n-C9, n-C10, He, Ar
# 对应索引 0-20

IDEAL_CP_COEFFS_MAP = {
    "methane": (-29776.4, 7.95454, 43.9417, 1037.09, 1.56373, 813.205, -24.9027, 1019.98, -10.1601, 1070.14, -20.0615),
    "nitrogen": (-3495.34, 6.95587, 0.272892, 662.738, -0.291318, -680.562, 1.78980, 1740.06, 0.0, 100.0, 4.49823),
    "carbon_dioxide": (20.7307, 6.96237, 2.68645, 500.371, -2.56429, -530.443, 3.91921, 500.198, 2.13290, 2197.22, 5.81381),
    "ethane": (-37524.4, 7.98139, 24.3668, 752.320, 3.53990, 272.846, 8.44724, 1020.13, -13.2732, 869.510, -22.4010),
    "propane": (-56072.1, 8.14319, 37.0629, 735.402, 9.38159, 247.190, 13.4556, 1454.78, -11.7342, 984.518, -24.0426),
    "water": (-13773.1, 7.97183, 6.27078, 2572.63, 2.05010, 1156.72, 0.0, 100.0, 0.0, 100.0, -3.24989),
    "hydrogen_sulfide": (-10085.4, 7.94680, -0.0838, 433.801, 2.85539, 843.792, 6.31595, 1481.43, -2.88457, 1102.23, -0.51551),
    "hydrogen": (-5565.6, 6.66789, 2.33458, 2584.98, 0.749019, 559.656, 0.0, 100.0, 0.0, 100.0, -7.94821),
    "carbon_monoxide": (-2753.49, 6.95854, 2.02441, 1541.22, 0.096774, 3674.81, 0.0, 100.0, 0.0, 100.0, 6.23387),
    "oxygen": (-3497.45, 6.96302, 2.40013, 2522.05, 2.21752, 1154.15, 0.0, 100.0, 0.0, 100.0, 9.19749),
    "isobutane": (-72387.0, 17.8143, 58.2062, 1787.39, 40.7621, 808.645, 0.0, 100.0, 0.0, 100.0, -44.1341),
    "n_butane": (-72674.8, 18.6383, 57.4178, 1792.73, 38.6599, 814.151, 0.0, 100.0, 0.0, 100.0, -46.1938),
    "isopentane": (-91505.5, 21.3861, 74.3410, 1701.58, 47.0587, 775.899, 0.0, 100.0, 0.0, 100.0, -60.2474),
    "n_pentane": (-83845.2, 22.5012, 69.5789, 1719.58, 46.2164, 802.174, 0.0, 100.0, 0.0, 100.0, -62.2197),
    "n_hexane": (-94982.5, 26.6225, 80.3819, 1718.49, 55.6598, 802.069, 0.0, 100.0, 0.0, 100.0, -77.5366),
    "n_heptane": (-103353.0, 30.4029, 90.6941, 1669.32, 63.2028, 786.001, 0.0, 100.0, 0.0, 100.0, -92.0164),
    "n_octane": (-109674.0, 34.0847, 100.253, 1611.55, 69.7675, 768.847, 0.0, 100.0, 0.0, 100.0, -106.149),
    "n_nonane": (-122599.0, 38.5014, 111.446, 1646.48, 80.5015, 781.588, 0.0, 100.0, 0.0, 100.0, -122.444),
    "n_decane": (-133564.0, 42.7143, 122.173, 1654.85, 90.2255, 785.564, 0.0, 100.0, 0.0, 100.0, -138.006),
    "helium": (0.0, 4.968, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 1.8198),
    "argon": (0.0, 4.968, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 8.6776),
}

# 映射内部索引到 Cp0 键名
INDEX_TO_CP0_KEY = [
    "methane", "nitrogen", "carbon_dioxide", "ethane", "propane", "water", "hydrogen_sulfide", "hydrogen",
    "carbon_monoxide", "oxygen", "isobutane", "n_butane", "isopentane", "n_pentane", "n_hexane",
    "n_heptane", "n_octane", "n_nonane", "n_decane", "helium", "argon"
]

CAL_TO_J = 4.184

def cp0_pure(component_idx: int, T: float) -> float:
    """纯组分理想气比定压热容 cp0(T)，J/(mol·K)"""
    key = INDEX_TO_CP0_KEY[component_idx]
    try:
        A, B, C, D, E, F, G, H, I, Jc, K_coeff = IDEAL_CP_COEFFS_MAP[key]
    except KeyError:
        return 0.0 # Should not happen if map is complete

    t1 = D / T
    t2 = F / T
    t3 = H / T
    t4 = Jc / T

    # 避免除零或溢出
    def safe_sinh(x): return math.sinh(x) if abs(x) < 700 else math.inf
    def safe_cosh(x): return math.cosh(x) if abs(x) < 700 else math.inf

    # 注意：当 T 很大或很小时，t 可能很大。
    # 这里假设 T 在合理范围内。
    
    term1 = C * (t1 / math.sinh(t1)) ** 2 if abs(t1) > 1e-9 else C
    term2 = E * (t2 / math.cosh(t2)) ** 2
    term3 = G * (t3 / math.sinh(t3)) ** 2 if abs(t3) > 1e-9 else G
    term4 = I * (t4 / math.cosh(t4)) ** 2

    cp_cal = B + term1 + term2 + term3 + term4
    return cp_cal * CAL_TO_J

def cp0_mixture(x: np.ndarray, T: float) -> float:
    """混合气理想气 cp0，J/(mol·K)"""
    cp_mix = 0.0
    for i in range(len(x)):
        if x[i] > 1e-9:
            cp_mix += x[i] * cp0_pure(i, T)
    return cp_mix

# ----------------- AGA8 核心逻辑复现 (支持 T, rho -> Z) -----------------

def calculate_Z_TP_rho(T: float, rho: float, x: np.ndarray) -> float:
    """
    根据 T (K), rho (mol/dm3), x 计算 Z
    逻辑复制自 aga8_calculator/calculator.py
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

    # Part 2: 计算 Cn 所需的中间变量
    F0 = np.sum(x**2 * F)
    Q0 = np.sum(x * Q)
    sum1_G = np.sum(x * G)
    sum2_E = np.sum(x * E**2.5)

    G0_term = np.triu(x_outer * (Gx - 1) * np.add.outer(G, G), k=1)
    G0 = sum1_G + np.sum(G0_term)

    U0_term = np.triu(x_outer * (Ux**5 - 1) * (np.outer(E, E)**2.5), k=1)
    U0 = (sum2_E**2 + np.sum(U0_term))**0.2

    # Part 3: 计算 K0
    sum1_K = np.sum(x * K**2.5)
    sum2_K_term = np.triu(x_outer * (Kx**5 - 1) * (np.outer(K, K)**2.5), k=1)
    K0 = (sum1_K**2 + 2 * np.sum(sum2_K_term))**0.2

    # Part 4: 计算 SUM1
    n_range_sum1 = np.arange(12, 18)
    Cn_vec_sum1 = a[n_range_sum1] * ((G0 + 1 - g[n_range_sum1])**g[n_range_sum1]) * \
        (((Q0**2) + 1 - q[n_range_sum1])**q[n_range_sum1]) * \
        ((F0 + 1 - f[n_range_sum1])**f[n_range_sum1]) * \
        (U0**u[n_range_sum1]) * (T**(-u[n_range_sum1]))
    SUM1 = np.sum(Cn_vec_sum1)

    # Part 5: 计算 P 和 Z
    # P = pm * R * T * (1 + B_calc * pm - pr * SUM1 + SUM2)
    # Z = P / (pm * R * T) = 1 + B_calc * pm - pr * SUM1 + SUM2
    
    pm = rho
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

    Z = 1 + B_calc * pm - pr * SUM1 + SUM2
    return Z

# ----------------- 导数与声速计算 -----------------

class AGA8Wrapper:
    def __init__(self, x):
        self.x = x

    def Z(self, T, rho):
        return calculate_Z_TP_rho(T, rho, self.x)

    def dZ_dT(self, T, rho, dT=1e-4):
        # 使用 5 点差分公式提高精度
        # f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
        return (-self.Z(T+2*dT, rho) + 8*self.Z(T+dT, rho) -
                 8*self.Z(T-dT, rho) + self.Z(T-2*dT, rho)) / (12*dT)

    def dZ_drho(self, T, rho, drho=None):
        if drho is None:
            drho = 1e-5 * max(1.0, rho)
        # 5 点差分
        return (-self.Z(T, rho+2*drho) + 8*self.Z(T, rho+drho) -
                 8*self.Z(T, rho-drho) + self.Z(T, rho-2*drho)) / (12*drho)
    
    def d2Z_dT2(self, T, rho, dT=1e-4):
        # 5 点差分求二阶导
        # f''(x) ≈ (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h^2)
        return (-self.Z(T+2*dT, rho) + 16*self.Z(T+dT, rho) -
                 30*self.Z(T, rho) + 16*self.Z(T-dT, rho) - self.Z(T-2*dT, rho)) / (12*dT*dT)

def _cv_integrand(T, rho, eos, dT_rel):
    # 减小 dT 以匹配高阶差分的需求
    dT = max(1e-4, dT_rel * T)
    dZdT = eos.dZ_dT(T, rho, dT=dT)
    d2ZdT2 = eos.d2Z_dT2(T, rho, dT=dT)
    return (T / rho) * d2ZdT2 + (2.0 / rho) * dZdT

def calculate_sound_speed_aga8(T_K, P_MPa, x_arr):
    """
    计算声速
    T_K: 温度 K
    P_MPa: 压力 MPa
    x_arr: 组分数组 (长度21)
    """
    # 1. 计算密度 (使用 aga8_calculator 包)
    from aga8_calculator.calculator import calculate_z_factor_bisection
    
    # 提高密度计算的收敛精度
    Z_eq, rho_molar, _, _, _ = calculate_z_factor_bisection(
        T_K, P_MPa, x_arr, tolerance=1e-9, max_iterations=2000
    )
    
    # 2. 准备计算导数
    eos = AGA8Wrapper(x_arr)
    
    # 3. 计算 Cv (积分)
    # R_si = 8.314462618... J/(mol K)
    R_si = 8.31446261815324
    
    cp0 = cp0_mixture(x_arr, T_K) # J/(mol K)
    
    # 增加积分步数
    n_int_steps = 200
    dT_rel = 1e-4
    
    integral = 0.0
    rho_prev = 0.0
    drho = rho_molar / n_int_steps
    
    # 使用 Simpson's Rule (辛普森积分法) 提高积分精度
    # 需要奇数个点 (n_int_steps 为偶数区间)
    # 这里简单起见，使用更密集的梯形法或中点法通常足够，
    # 但为了追求极致，我们用简单的密集网格求和。
    
    for i in range(1, n_int_steps+1):
        rho_i = i * drho
        rho_mid = 0.5 * (rho_prev + rho_i)
        rho_mid = max(rho_mid, 1e-9)
        integrand = _cv_integrand(T_K, rho_mid, eos, dT_rel)
        integral += integrand * (rho_i - rho_prev)
        rho_prev = rho_i
        
    cv = cp0 - R_si * (1.0 + T_K * integral)
    
    # 4. 计算 Cp
    # 需要 dZ/dT, dZ/drho
    
    dZdT = eos.dZ_dT(T_K, rho_molar, dT=max(1e-4, dT_rel*T_K))
    dZdrho = eos.dZ_drho(T_K, rho_molar)
    
    # 公式: Cp = Cv + R * (Z + T*dZdT)^2 / (Z + rho*dZdrho)
    # 所有项都是无量纲或 J/(mol K)
    # rho * dZdrho 是无量纲的
    
    numerator = (Z_eq + T_K * dZdT)**2
    denominator = Z_eq + rho_molar * dZdrho
    
    cp = cv + R_si * numerator / denominator
    
    # 5. 计算声速
    # W = sqrt( (Cp/Cv) * (R_si * T / M_kg) * (Z + rho*dZdrho) )
    # M 需要转换为 kg/mol
    
    M_mix_g = np.sum(x_arr * M) # g/mol
    M_mix_kg = M_mix_g / 1000.0
    
    factor = (cp / cv) * (R_si * T_K / M_mix_kg) * denominator
    
    if factor < 0:
        return 0.0
        
    w = math.sqrt(factor)
    
    return w, cv, cp, rho_molar, Z_eq

# ----------------- 测试 -----------------

if __name__ == "__main__":
    # ----------------- 测试用例 1: 纯甲烷 -----------------
    print("=" * 60)
    print("Test Case 1: Pure Methane (20 C, 8 MPa)")
    x1 = np.zeros(N)
    x1[0] = 1.0 # Methane
    
    w1, cv1, cp1, rho1, Z1 = calculate_sound_speed_aga8(293.15, 8.0, x1)
    print(f"  Z           = {Z1:.8f} (Ref: 0.865613)")
    print(f"  Sound Speed = {w1:.4f} m/s (Ref: 432.944)")
    
    # ----------------- 测试用例 2: 21种组分混合物 (表 A.2 / A.1.3) -----------------
    print("\n" + "=" * 60)
    print("Test Case 2: 21-Component Mixture (40 C, 6 MPa)")
    
    # 组分顺序 (aga8_calculator.constants / router.py):
    # 0:CH4, 1:N2, 2:CO2, 3:C2H6, 4:C3H8, 5:H2O, 6:H2S, 7:H2, 8:CO, 9:O2,
    # 10:i-C4, 11:n-C4, 12:i-C5, 13:n-C5, 14:n-C6, 15:n-C7, 16:n-C8, 17:n-C9, 18:n-C10,
    # 19:He, 20:Ar
    
    x2 = np.zeros(N)
    # 输入摩尔分数 (%)
    x2[0] = 86.29  # Methane
    x2[1] = 2.0    # Nitrogen
    x2[2] = 0.50   # Carbon Dioxide
    x2[3] = 5.0    # Ethane
    x2[4] = 3.0    # Propane
    x2[5] = 0.01   # Water
    x2[6] = 0.1    # Hydrogen Sulfide
    x2[7] = 0.01   # Hydrogen
    x2[8] = 0.01   # Carbon Monoxide
    x2[9] = 0.02   # Oxygen
    x2[10] = 1.10  # Isobutane
    x2[11] = 0.90  # n-Butane
    x2[12] = 0.35  # Isopentane
    x2[13] = 0.25  # n-Pentane
    x2[14] = 0.20  # n-Hexane
    x2[15] = 0.10  # n-Heptane
    x2[16] = 0.05  # n-Octane
    x2[17] = 0.01  # n-Nonane
    x2[18] = 0.02  # n-Decane
    x2[19] = 0.04  # Helium
    x2[20] = 0.04  # Argon
    
    # 归一化 (输入是百分比)
    x2 = x2 / 100.0
    
    T_C_2 = 40.0
    P_MPa_2 = 6.000
    T_K_2 = T_C_2 + 273.15
    
    w2, cv2, cp2, rho2, Z2 = calculate_sound_speed_aga8(T_K_2, P_MPa_2, x2)
    
    # 计算摩尔质量用于对比 (g/mol)
    M_mix_g = np.sum(x2 * M)
    
    # 单位转换用于对比
    # Cp, Cv: J/(mol K) -> kJ/(kg K)
    # Factor: (J/mol K) / (g/mol) = (J/g K) = (kJ/kg K)
    cp2_mass = cp2 / M_mix_g
    cv2_mass = cv2 / M_mix_g
    
    print(f"{'Property':<20} | {'Reference':<15} | {'Calculated':<15} | {'Error %':<10}")
    print("-" * 70)
    
    ref2 = {
        "rho_molar": 2.62533592, # mol/dm3
        "M": 19.4780144,         # g/mol
        "Z": 0.877763047,
        "speed_sound": 391.528389, # m/s
        "cp_mass": 2.55641833,   # kJ/(kg K)
        "cv_mass": 1.73699984    # kJ/(kg K)
    }
    
    calc2 = {
        "rho_molar": rho2,
        "M": M_mix_g,
        "Z": Z2,
        "speed_sound": w2,
        "cp_mass": cp2_mass,
        "cv_mass": cv2_mass
    }
    
    for key, val_ref in ref2.items():
        val_calc = calc2[key]
        err = abs(val_calc - val_ref) / val_ref * 100
        print(f"{key:<20} | {val_ref:<15.8f} | {val_calc:<15.8f} | {err:<10.4f}")

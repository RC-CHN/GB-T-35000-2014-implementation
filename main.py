import math
from dataclasses import dataclass
from typing import Dict
import numpy as np
from pyaga8 import AGA8Detail

# 常数
R = 8.31446261815324  # J/(mol·K) 通用气体常数
CAL_TO_J = 4.184  # 1 cal = 4.184 J

# 表 A.1 / AGA‑10 Table 2 的理想气 cp0 系数（纯组分）
IDEAL_CP_COEFFS: Dict[str, tuple] = {
    "methane": (
        -29776.4,
        7.95454,
        43.9417,
        1037.09,
        1.56373,
        813.205,
        -24.9027,
        1019.98,
        -10.1601,
        1070.14,
        -20.0615,
    ),
    "nitrogen": (
        -3495.34,
        6.95587,
        0.272892,
        662.738,
        -0.291318,
        -680.562,
        1.78980,
        1740.06,
        0.0,
        100.0,
        4.49823,
    ),
    "carbon_dioxide": (
        20.7307,
        6.96237,
        2.68645,
        500.371,
        -2.56429,
        -530.443,
        3.91921,
        500.198,
        2.13290,
        2197.22,
        5.81381,
    ),
    "ethane": (
        -37524.4,
        7.98139,
        24.3668,
        752.320,
        3.53990,
        272.846,
        8.44724,
        1020.13,
        -13.2732,
        869.510,
        -22.4010,
    ),
    "propane": (
        -56072.1,
        8.14319,
        37.0629,
        735.402,
        9.38159,
        247.190,
        13.4556,
        1454.78,
        -11.7342,
        984.518,
        -24.0426,
    ),
    "water": (
        -13773.1,
        7.97183,
        6.27078,
        2572.63,
        2.05010,
        1156.72,
        0.0,
        100.0,
        0.0,
        100.0,
        -3.24989,
    ),
    "hydrogen_sulfide": (
        -10085.4,
        7.94680,
        -0.0838,
        433.801,
        2.85539,
        843.792,
        6.31595,
        1481.43,
        -2.88457,
        1102.23,
        -0.51551,
    ),
    "hydrogen": (
        -5565.6,
        6.66789,
        2.33458,
        2584.98,
        0.749019,
        559.656,
        0.0,
        100.0,
        0.0,
        100.0,
        -7.94821,
    ),
    "carbon_monoxide": (
        -2753.49,
        6.95854,
        2.02441,
        1541.22,
        0.096774,
        3674.81,
        0.0,
        100.0,
        0.0,
        100.0,
        6.23387,
    ),
    "oxygen": (
        -3497.45,
        6.96302,
        2.40013,
        2522.05,
        2.21752,
        1154.15,
        0.0,
        100.0,
        0.0,
        100.0,
        9.19749,
    ),
    "isobutane": (
        -72387.0,
        17.8143,
        58.2062,
        1787.39,
        40.7621,
        808.645,
        0.0,
        100.0,
        0.0,
        100.0,
        -44.1341,
    ),
    "n_butane": (
        -72674.8,
        18.6383,
        57.4178,
        1792.73,
        38.6599,
        814.151,
        0.0,
        100.0,
        0.0,
        100.0,
        -46.1938,
    ),
    "isopentane": (
        -91505.5,
        21.3861,
        74.3410,
        1701.58,
        47.0587,
        775.899,
        0.0,
        100.0,
        0.0,
        100.0,
        -60.2474,
    ),
    "n_pentane": (
        -83845.2,
        22.5012,
        69.5789,
        1719.58,
        46.2164,
        802.174,
        0.0,
        100.0,
        0.0,
        100.0,
        -62.2197,
    ),
    "n_hexane": (
        -94982.5,
        26.6225,
        80.3819,
        1718.49,
        55.6598,
        802.069,
        0.0,
        100.0,
        0.0,
        100.0,
        -77.5366,
    ),
    "n_heptane": (
        -103353.0,
        30.4029,
        90.6941,
        1669.32,
        63.2028,
        786.001,
        0.0,
        100.0,
        0.0,
        100.0,
        -92.0164,
    ),
    "n_octane": (
        -109674.0,
        34.0847,
        100.253,
        1611.55,
        69.7675,
        768.847,
        0.0,
        100.0,
        0.0,
        100.0,
        -106.149,
    ),
    "n_nonane": (
        -122599.0,
        38.5014,
        111.446,
        1646.48,
        80.5015,
        781.588,
        0.0,
        100.0,
        0.0,
        100.0,
        -122.444,
    ),
    "n_decane": (
        -133564.0,
        42.7143,
        122.173,
        1654.85,
        90.2255,
        785.564,
        0.0,
        100.0,
        0.0,
        100.0,
        -138.006,
    ),
    "helium": (0.0, 4.968, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 1.8198),
    "argon": (0.0, 4.968, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 8.6776),
}

# ---------- 理想气 cp0 ----------


def cp0_pure(component: str, T: float) -> float:
    """计算纯组分理想气比定压热容 cp0(T)，单位 J/(mol·K)"""
    try:
        A, B, C, D, E, F, G, H, I, Jc, K = IDEAL_CP_COEFFS[component]
    except KeyError:
        raise KeyError(f"Unknown component {component!r}")

    t1 = D / T
    t2 = F / T
    t3 = H / T
    t4 = Jc / T

    cp_cal = (
        B
        + C * (t1 / math.sinh(t1)) ** 2
        + E * (t2 / math.cosh(t2)) ** 2
        + G * (t3 / math.sinh(t3)) ** 2
        + I * (t4 / math.cosh(t4)) ** 2
    )
    return cp_cal * CAL_TO_J


def cp0_mixture(x: Dict[str, float], T: float) -> float:
    """计算混合气体理想气 cp0，摩尔分数加权"""
    return sum(x_i * cp0_pure(comp, T) for comp, x_i in x.items())


# ---------- AGA8 声速计算 ----------


class AGA8EOS:
    """基于 pyaga8 库的 AGA8 方程 EOS"""

    def __init__(self):
        self.model = AGA8Detail()  # 使用 pyaga8 中的 AGA8 详细模型

    def Z(self, T: float, rho: float, x: Dict[str, float]) -> float:
        """根据 T, rho, x 返回压缩因子 Z"""
        x_vector = [x.get(comp, 0) for comp in self.model.components]
        self.model.set_state(T, rho, x_vector)  # 设置状态
        return self.model.Z


# ---------- cv、cp 计算 ----------


def _cv_integrand(
    T: float, rho: float, x: Dict[str, float], eos: AGA8EOS, dT_rel: float
) -> float:
    """计算积分项"""
    dT = max(1e-3, dT_rel * T)
    dZdT = eos.dZ_dT(T, rho, x, dT=dT)
    d2ZdT2 = eos.d2Z_dT2(T, rho, x, dT=dT)
    return (T / rho) * d2ZdT2 + 2.0 / rho * dZdT


def cv_real(
    T: float,
    p: float,
    x: Dict[str, float],
    eos: AGA8EOS,
    n_int_steps: int = 80,
    dT_rel: float = 1e-3,
):
    """计算 cv，返回 cv, rho, Z"""
    rho = eos.density(p, T, x)
    cp0 = cp0_mixture(x, T)

    # 对 ρ 做数值积分
    integral = 0.0
    rho_prev = 0.0
    drho = rho / n_int_steps
    for i in range(1, n_int_steps + 1):
        rho_i = i * drho
        rho_mid = 0.5 * (rho_prev + rho_i)
        rho_mid = max(rho_mid, 1e-9)
        integrand = _cv_integrand(T, rho_mid, x, eos, dT_rel)
        integral += integrand * (rho_i - rho_prev)
        rho_prev = rho_i

    cv = cp0 - R * (1.0 + T * integral)
    Z = eos.Z(T, rho, x)
    return cv, rho, Z


# ---------- 声速计算 ----------


def sound_speed(
    T: float,
    p: float,
    x: Dict[str, float],
    M: float,
    eos: AGA8EOS,
    n_int_steps: int = 80,
    dT_rel: float = 1e-3,
) -> float:
    """计算声速 W"""
    cp, cv, rho, Z, dZdT, dZdrho = cv_real(
        T, p, x, eos, n_int_steps=n_int_steps, dT_rel=dT_rel
    )

    factor = (cp / cv) * (R * T / M) * (Z + rho * dZdrho)
    if factor < 0.0:
        raise ValueError(f"Negative value under sqrt for sound speed: {factor}")
    return math.sqrt(factor)


# ---------- 测试部分 ----------

if __name__ == "__main__":
    x_test = {"methane": 1.0}
    T_test = 300.0  # K
    p_test = 101_325.0  # Pa
    M_CH4 = 16.043e-3  # kg/mol

    eos = AGA8EOS()  # 使用 AGA8 模型
    w = sound_speed(T_test, p_test, x_test, M_CH4, eos)

    cp0 = cp0_mixture(x_test, T_test)
    cv_expected = cp0 - R
    cp_expected = cp0
    w_expected = math.sqrt((cp_expected / cv_expected) * (R * T_test / M_CH4))

    print("cp0 (CH4, ideal)  =", cp0)
    print("cv  (CH4, ideal)  =", cv_expected)
    print("W (code)         =", w)
    print("W (analytical)   =", w_expected)

import math
from dataclasses import dataclass
from typing import Dict
import numpy as np
import CoolProp.CoolProp as CP

# ----------------- 常数 -----------------
R_u = 8.31446261815324  # J/(mol·K) 通用气体常数
CAL_TO_J = 4.184  # 1 cal = 4.184 J

# ----------------- 理想气 cp0 系数 -----------------
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
    # 这里只写 methane，只需验证示例
}


def cp0_pure(component: str, T: float) -> float:
    """纯组分理想气比定压热容 cp0(T)，J/(mol·K)"""
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
    """混合气理想气 cp0，J/(mol·K)"""
    return sum(x_i * cp0_pure(comp, T) for comp, x_i in x.items())


# ----------------- CoolProp EOS Wrapper -----------------


class CoolPropEOS:
    """用 CoolProp 计算压缩因子 Z 及相关导数"""

    def __init__(self, fluid):
        self.fluid = fluid

    def Z(self, T, rho_molar):
        """压缩因子 Z = p / (rho*R*T)"""
        p = CP.PropsSI("P", "T", T, "Dmolar", rho_molar, self.fluid)
        return p / (rho_molar * R_u * T)

    def dZ_dT(self, T, rho_molar, dT=1e-3):
        """数值差分求 dZ/dT |_ρ"""
        return (self.Z(T + dT, rho_molar) - self.Z(T - dT, rho_molar)) / (2 * dT)

    def dZ_drho(self, T, rho_molar, drho=None):
        """数值差分求 dZ/dρ |_T"""
        if drho is None:
            drho = 1e-6 * max(1.0, rho_molar)
        return (self.Z(T, rho_molar + drho) - self.Z(T, rho_molar - drho)) / (2 * drho)

    def density(self, p, T, x):
        """由 p,T 解 molar density: p = rho*R*T*Z  => 迭代求 rho"""
        # 初猜:
        rho = p / (R_u * T)
        for _ in range(30):
            Z = CP.PropsSI("Z", "T", T, "P", p, self.fluid)
            f = rho * R_u * T * Z - p
            if abs(f) < 1e-9 * p:
                return rho
            # 用 CoolProp 计算导数 近似
            drho = 1e-6 * max(rho, 1.0)
            dZdrho = self.dZ_drho(T, rho, drho)
            fp = R_u * T * (Z + rho * dZdrho)
            rho -= f / fp
        return rho


# ----------------- cv 计算 -----------------


def _cv_integrand(T, rho_molar, x, eos, dT_rel):
    dT = max(1e-3, dT_rel * T)
    dZdT = eos.dZ_dT(T, rho_molar, dT=dT)
    # 二阶导同样数值差分
    d2ZdT2 = (
        eos.Z(T + dT, rho_molar) - 2 * eos.Z(T, rho_molar) + eos.Z(T - dT, rho_molar)
    ) / (dT * dT)
    return (T / rho_molar) * d2ZdT2 + (2.0 / rho_molar) * dZdT


def cv_real(T, p, x, eos, n_int_steps=80, dT_rel=1e-3):
    """计算真实气体 cv"""
    # molar density (mol/m3)
    rho_molar = CP.PropsSI("Dmolar", "T", T, "P", p, eos.fluid)
    cp0 = cp0_mixture(x, T)

    # 积分 ρ -> 0..rho
    integral = 0.0
    rho_prev = 0.0
    drho = rho_molar / n_int_steps
    for i in range(1, n_int_steps + 1):
        rho_i = i * drho
        rho_mid = 0.5 * (rho_prev + rho_i)
        rho_mid = max(rho_mid, 1e-9)
        integrand = _cv_integrand(T, rho_mid, x, eos, dT_rel)
        integral += integrand * (rho_i - rho_prev)
        rho_prev = rho_i

    cv = cp0 - R_u * (1.0 + T * integral)
    Z = CP.PropsSI("Z", "T", T, "P", p, eos.fluid)
    # dZ/dT, dZ/drho
    dZdT = eos.dZ_dT(T, rho_molar, dT=max(1e-3, dT_rel * T))
    dZdrho = eos.dZ_drho(T, rho_molar)
    return cv, Z, dZdT, dZdrho, rho_molar


# ----------------- 声速 计算 -----------------


def sound_speed(T, p, x, M, eos):
    """理论声速 W"""
    cv, Z, dZdT, dZdrho, rho_molar = cv_real(T, p, x, eos)
    cp = cv + R_u * (Z + T * dZdT) ** 2 / (Z + rho_molar * dZdrho)
    factor = (cp / cv) * (R_u * T / M) * (Z + rho_molar * dZdrho)
    return math.sqrt(factor)


# ----------------- 主程序验证 -----------------

if __name__ == "__main__":
    fluid = "Methane"

    # 输入条件 (来自截图)
    T_C = 20.0
    P_MPa = 8.000

    T = T_C + 273.15  # K
    p = P_MPa * 1e6  # Pa

    print(f"Input:")
    print(f"  Temperature: {T_C} C ({T} K)")
    print(f"  Pressure:    {P_MPa} MPa ({p} Pa)")
    print(f"  Fluid:       {fluid}")
    print("-" * 85)

    # 参考值 (来自截图)
    ref = {
        "rho_molar": 3.79174963,  # mol/dm3
        "M": 16.0430000,  # g/mol
        "Z": 0.865613011,
        "cp0_mass": 2.21437395,  # kJ/(kg K)
        "cp_mass": 2.86913018,  # kJ/(kg K)
        "cv_mass": 1.78350108,  # kJ/(kg K)
        "speed_sound": 432.944437,  # m/s
    }

    # CoolProp 计算
    # 注意: CoolProp 使用 SI 单位 (mol/m3, kg/mol, J/kg/K, Pa, K)

    rho_molar_cp = CP.PropsSI("Dmolar", "T", T, "P", p, fluid)  # mol/m3
    M_cp = CP.PropsSI("M", fluid)  # kg/mol
    Z_cp = CP.PropsSI("Z", "T", T, "P", p, fluid)

    # 理想气体热容 Cp0
    cp0_mass_cp = CP.PropsSI("Cp0mass", "T", T, "P", p, fluid)  # J/kg/K

    # 真实气体热容
    cp_mass_cp = CP.PropsSI("Cpmass", "T", T, "P", p, fluid)  # J/kg/K
    cv_mass_cp = CP.PropsSI("Cvmass", "T", T, "P", p, fluid)  # J/kg/K

    # 声速
    speed_sound_cp = CP.PropsSI("A", "T", T, "P", p, fluid)  # m/s

    # 单位转换用于对比
    calc = {
        "rho_molar": rho_molar_cp / 1000.0,  # mol/m3 -> mol/dm3
        "M": M_cp * 1000.0,  # kg/mol -> g/mol
        "Z": Z_cp,
        "cp0_mass": cp0_mass_cp / 1000.0,  # J -> kJ
        "cp_mass": cp_mass_cp / 1000.0,  # J -> kJ
        "cv_mass": cv_mass_cp / 1000.0,  # J -> kJ
        "speed_sound": speed_sound_cp,
    }

    print(
        f"{'Property':<20} | {'Reference':<15} | {'Calculated (CP)':<18} | {'Error %':<10}"
    )
    print("-" * 85)

    for key, val_ref in ref.items():
        val_calc = calc[key]
        error = abs(val_calc - val_ref) / val_ref * 100
        print(f"{key:<20} | {val_ref:<15.8f} | {val_calc:<18.8f} | {error:<10.4f}")

    print("-" * 85)

    # 验证手动实现的 sound_speed 函数
    print("\nVerifying manual implementation in test.py:")
    x = {"methane": 1.0}
    eos = CoolPropEOS(fluid)
    M_si = M_cp  # kg/mol

    w_manual = sound_speed(T, p, x, M_si, eos)
    print(f"Manual sound_speed : {w_manual:.6f} m/s")
    print(f"Reference          : {ref['speed_sound']:.6f} m/s")
    print(
        f"Error              : {abs(w_manual - ref['speed_sound']) / ref['speed_sound'] * 100:.4f} %"
    )

# -*- coding: utf-8 -*-
# AGA8-92DC Calculator API Router

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict

# 从当前模块导入核心计算函数和常量
from .calculator import calculate_z_factor_bisection
from .constants import N

# --- API Router 定义 ---
router = APIRouter()

# --- 权威的内部映射与常量 ---

# 映射1: 将API接收的名称(CoolProp/refer格式)映射到我们项目内部的组分名
API_TO_INTERNAL_NAME_MAP = {
    'Methane': 'CH4', 'Nitrogen': 'N2', 'CarbonDioxide': 'CO2', 'Ethane': 'C2H6',
    'Propane': 'C3H8', 'Water': 'H2O', 'HydrogenSulfide': 'H2S', 'Hydrogen': 'H2',
    'CarbonMonoxide': 'CO', 'Oxygen': 'O2', 'Isobutane': 'i-C4H10', 'Butane': 'n-C4H10',
    'Isopentane': 'i-C5H12', 'Pentane': 'n-C5H12', 'Hexane': 'n-C6H14',
    'Heptane': 'n-C7H16', 'Octane': 'n-C8H18', 'Nonane': 'n-C9H20',
    'Decane': 'n-C10H22', 'Helium': 'He', 'Argon': 'Ar'
}

# 映射2: 我们项目内部组分名到其在Numpy数组中固定索引的映射
INTERNAL_NAME_TO_INDEX_MAP = {
    "CH4": 0, "N2": 1, "CO2": 2, "C2H6": 3, "C3H8": 4, "H2O": 5, "H2S": 6, "H2": 7,
    "CO": 8, "O2": 9, "i-C4H10": 10, "n-C4H10": 11, "i-C5H12": 12, "n-C5H12": 13,
    "n-C6H14": 14, "n-C7H16": 15, "n-C8H18": 16, "n-C9H20": 17, "n-C10H22": 18,
    "He": 19, "Ar": 20
}

# --- API 模型定义 (与 gerg_calculator 兼容) ---


class CalculationRequest(BaseModel):
    base_components: Dict[str, float] = Field(..., example={
                                              "Methane": 0.9, "Nitrogen": 0.05, "Ethane": 0.05})
    hydrogen_fraction: float = Field(
        0.0, ge=0.0, le=1.0, description="氢气的摩尔分数")
    T: float = Field(..., example=288.15, description="温度 (K)")
    P_kPa: float = Field(..., example=1013.25, description="压力 (kPa)")


class CalculationResponse(BaseModel):
    final_components: Dict[str, float]
    compression_factor: float

# --- 内部辅助函数 ---


def adjust_compositions_with_hydrogen(base_components: Dict[str, float], hydrogen_fraction: float) -> Dict[str, float]:
    """根据氢气含量调整组分，并返回以API标准名称为键的字典。"""
    if not (0.0 <= hydrogen_fraction <= 1.0):
        raise ValueError("氢气摩尔分数必须在 0.0 和 1.0 之间。")

    final_components = {}

    base_total = sum(base_components.values())
    if base_total > 0:
        scale_factor = (1.0 - hydrogen_fraction) / base_total
        for comp, frac in base_components.items():
            final_components[comp] = frac * scale_factor
    elif hydrogen_fraction < 1.0 and len(base_components) > 0:
        raise ValueError("基础组分摩尔分数总和为0，无法在添加氢气后进行归一化。")

    if hydrogen_fraction > 0:
        final_components['Hydrogen'] = hydrogen_fraction

    total_sum = sum(final_components.values())
    if total_sum > 0 and abs(total_sum - 1.0) > 1e-9:
        for comp in final_components:
            final_components[comp] /= total_sum

    return final_components

# --- API 端点定义 ---


@router.post("/calculate", response_model=CalculationResponse)
def calculate(request: CalculationRequest):
    """
    计算给定组分、温度和压力下的气体压缩因子。
    该接口在内部将API格式的输入转换为本地AGA8计算引擎所需的格式。
    """
    try:
        final_components_api_names = adjust_compositions_with_hydrogen(
            request.base_components, request.hydrogen_fraction
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    x = np.zeros(N)
    for api_name, fraction in final_components_api_names.items():
        internal_name = API_TO_INTERNAL_NAME_MAP.get(api_name)
        if internal_name is None:
            raise HTTPException(
                status_code=400, detail=f"不支持的组分名称: '{api_name}'")

        index = INTERNAL_NAME_TO_INDEX_MAP.get(internal_name)
        if index is None:
            raise HTTPException(
                status_code=500, detail=f"内部错误: 组分'{internal_name}'在顺序列表中未找到。")

        x[index] = fraction

    if abs(np.sum(x) - 1.0) > 1e-9:
        x /= np.sum(x)

    pressure_mpa = request.P_kPa / 1000.0

    try:
        Z, _, _, _, _ = calculate_z_factor_bisection(
            T=request.T, P0=pressure_mpa, x=x)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"核心计算模块发生错误: {e}")

    return CalculationResponse(
        final_components=final_components_api_names,
        compression_factor=Z,
    )

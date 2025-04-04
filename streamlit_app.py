# -*- coding: utf-8 -*-
"""
AR(3)模型PACF可视化教学工具
作者：Fanyin@swufe
最后更新：2025-4-4
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf, pacf

# --------------------------
# 显示自相关与偏自相关公式表
# --------------------------


def display_formula_table():
    """展示自相关与偏自相关的公式对比"""
    st.markdown("### 自相关系数与偏自相关系数的定义差异")

    st.latex(r"""
    \begin{array}{c|c|c}
    \textbf{阶数} & \textbf{自相关系数} \ \rho_k & \textbf{偏自相关系数} \ \phi_{kk} \\ \hline
    1 &\rho_1 = \mathrm{cor}(X_t, X_{t-1}) & \phi_{11} = \mathrm{cor}(X_t, X_{t-1}| \varnothing) \\
    2 &\rho_2 = \mathrm{cor}(X_t, X_{t-2}) & \phi_{22} = \mathrm{cor}(X_t, X_{t-2} | X_{t-1}) \\
    3 & \rho_3 = \mathrm{cor}(X_t, X_{t-3}) & \phi_{33} = \mathrm{cor}(X_t, X_{t-3} | X_{t-1}, X_{t-2}) \\
    \cdots & \cdots 
    \end{array}
    """)


# --------------------------
# 时间序列生成函数
# --------------------------
def generate_ar3_process(phi1, phi2, phi3, n):
    """生成平稳AR(3)序列"""
    # 平稳性检查
    coefficients = [1, -phi1, -phi2, -phi3]
    roots = np.roots(coefficients)
    if not all(np.abs(roots) < 1):
        raise ValueError("非平稳参数！")

    X = np.zeros(n + 3)
    epsilon = np.random.normal(0, 0.5, n + 3)
    for t in range(3, n + 3):
        X[t] = phi1 * X[t - 1] + phi2 * X[t - 2] + phi3 * X[t - 3] + epsilon[t]
    return X[3:]


def theoretical_pacf(phi1, phi2, phi3):
    """精确计算前三阶PACF"""
    acf1 = (phi1 + phi2 * phi3) / (1 - phi2 - phi1 * phi3 - phi3**2)
    acf2 = (phi1 + phi3) * acf1 + phi2

    # 计算PACF(1-3)
    pacf1 = acf1
    denominator = 1 - acf1**2
    pacf2 = (acf2 - acf1**2) / denominator if denominator > 1e-8 else np.nan
    pacf3 = phi3  # 直接取AR(3)系数

    return [pacf1, pacf2, pacf3]


# --------------------------
# ACF计算部分
# --------------------------
def calculate_acf(X, max_lag=3):
    """计算各阶自相关系数"""
    acf_values = []
    for lag in range(1, max_lag + 1):
        if len(X) <= lag:
            corr = np.nan
        else:
            corr = np.corrcoef(X[:-lag], X[lag:])[0, 1]
        acf_values.append(corr)
    return acf_values


# 在数据生成部分调用
def manual_pacf(X, max_lag=3):
    """手动计算偏自相关系数和残差数据"""
    pacf_values = []
    residual_data = pd.DataFrame()
    for lag in range(1, max_lag + 1):
        df = pd.DataFrame({'X_t': X[lag:]})
        for i in range(1, lag + 1):
            df[f'X_t-{i}'] = X[lag - i:-i]

        if lag > 1:
            predictors = [f'X_t-{i}' for i in range(1, lag)]
            model1 = LinearRegression()
            model1.fit(df[predictors], df['X_t'])
            resid_Xt = df['X_t'] - model1.predict(df[predictors])

            model2 = LinearRegression()
            model2.fit(df[predictors], df[f'X_t-{lag}'])
            resid_Xt_lag = df[f'X_t-{lag}'] - model2.predict(df[predictors])

        else:
            resid_Xt = df['X_t']
            resid_Xt_lag = df['X_t-1']

        residual_data[f'resid_Xt{lag}'] = resid_Xt.reset_index(drop=True)
        residual_data[f'resid_Xt_lag{lag}'] = resid_Xt_lag.reset_index(
            drop=True)

        pacf_values.append(np.corrcoef(resid_Xt, resid_Xt_lag)[0, 1])
    return pacf_values, residual_data


def validate_ml_input(X, y):
    """确保机器学习模型的输入数据有效"""
    X = np.asarray(X)
    y = np.asarray(y)

    # 双重检查NaN和inf
    mask = ~(np.isnan(X) | np.isinf(X) | np.isnan(y) | np.isinf(y))
    X = X[mask].reshape(-1, 1)
    y = y[mask]

    if len(X) < 2:
        raise ValueError("有效样本量不足，至少需要2个有效数据点")

    return X, y


# --------------------------
# 可视化函数
# --------------------------
def create_scatter_plots(raw_data, resid_data, lag, pacf_value, acf_value):
    """生成两个散点图：原始数据和残差数据"""

    # 设置画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 原始数据散点图（ACF）
    axes[0].scatter(raw_data[f'X_t-{lag}'], raw_data['X_t'], alpha=0.5)
    axes[0].set_title(f"Raw Correlation (ACF({lag})): {acf_value:.2f}")
    axes[0].set_xlabel(f"$X_{{t-{lag}}}$")
    axes[0].set_ylabel("$X_t$")
    axes[0].grid()

    # 原始数据拟合直线
    reg1 = LinearRegression()
    reg1.fit(raw_data[[f'X_t-{lag}']], raw_data['X_t'])
    x_vals = np.linspace(raw_data[f'X_t-{lag}'].min(),
                         raw_data[f'X_t-{lag}'].max(), 100)
    y_vals = reg1.predict(x_vals.reshape(-1, 1))
    axes[0].plot(x_vals, y_vals, color='red', alpha=0.8)

    # 残差数据散点图（PACF）
    axes[1].scatter(resid_data[f'resid_Xt{lag}'],
                    resid_data[f'resid_Xt_lag{lag}'],
                    alpha=0.5)
    axes[1].set_title(f"Partial Correlation (PACF({lag})): {pacf_value:.2f}")
    axes[1].set_xlabel(f"Residual for $X_{{t-{lag}}}$")
    axes[1].set_ylabel(f"Residual for $X_t$")
    axes[1].grid()

    # 残差数据拟合直线
    try:
        # 严格验证输入数据
        X, y = validate_ml_input(resid_data[f'resid_Xt_lag{lag}'],
                                 resid_data[f'resid_Xt{lag}'])

        reg2 = LinearRegression()
        reg2.fit(X, y)

        x_vals = np.linspace(X.min(), X.max(), 100)
        y_vals = reg2.predict(x_vals.reshape(-1, 1))
        axes[1].plot(x_vals, y_vals, color='red', alpha=0.8)

    except Exception as e:
        axes[1].text(0.5,
                     0.5,
                     f"绘图失败: {str(e)}",
                     ha='center',
                     va='center',
                     color='red')

    return fig


def create_comparison_table(ar_coefs, sample_data, manual_pacf):
    """生成三列对比表格"""
    # 计算理论值
    theory_vals = theoretical_pacf(*ar_coefs)

    # 计算statsmodels值
    try:
        sm_pacf = pacf(sample_data, nlags=3)[1:4]
    except:
        sm_pacf = [np.nan] * 3

    # 构建DataFrame
    df = pd.DataFrame({
        '滞后阶数': [1, 2, 3],
        '理论公式计算': np.round(theory_vals, 3),
        'statsmodels': np.round(sm_pacf, 3),
        '手动回归法': np.round(manual_pacf, 3)
    })

    # 添加差异高亮
    def highlight(row):
        colors = [''] * len(row)
        if abs(row['理论公式计算'] - row['statsmodels']) > 0.01:
            colors[2] = 'background-color: #ffe6e6'
        if abs(row['理论公式计算'] - row['手动回归法']) > 0.01:
            colors[3] = 'background-color: #e6f3ff'
        return colors

    return df.style.apply(highlight, axis=1)


# --------------------------
# Streamlit布局
# --------------------------

# 页面设置
st.set_page_config(page_title="AR(3) PACF教学工具", layout="wide")

st.title("📊 AR(3) 模型偏自相关系数(PACF)可视化工具")

st.sidebar.header("⚙️ 模型参数设置")

# 参数输入
phi1 = st.sidebar.slider("φ₁ (滞后 1 阶系数)", -1.0, 1.0, 0.7, step=0.05)
phi2 = st.sidebar.slider("φ₂ (滞后 2 阶系数)", -1.0, 1.0, -0.4, step=0.05)
phi3 = st.sidebar.slider("φ₃ (滞后 3 阶系数)", -1.0, 1.0, 0.2, step=0.05)
n_samples = st.sidebar.number_input("样本数量",
                                    min_value=100,
                                    max_value=5000,
                                    value=500,
                                    step=100)

# 数据生成与计算
ar3_data = generate_ar3_process(phi1, phi2, phi3, n_samples)
manual_pacf_values, residual_df = manual_pacf(ar3_data)
acf_values = calculate_acf(ar3_data)  # 计算ACF

st.subheader("AR(3) 时间序列可视化")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ar3_data, color='steelblue', linewidth=0.8, alpha=0.8)
ax.set_title(f"Generated AR(3) Series: φ1={phi1}, φ2={phi2}, φ3={phi3}",
             fontsize=12)
ax.set_xlabel("Time Step")
ax.set_ylabel("Value")
ax.grid(alpha=0.3)
st.pyplot(fig)

display_formula_table()

# 可视化
st.subheader("ACF值与PACF值对比")
for lag in range(1, 4):
    # 准备原始数据
    raw_data = pd.DataFrame({
        'X_t': ar3_data[lag:],
        f'X_t-{lag}': ar3_data[:-lag]
    })

    # 绘制图表
    fig = create_scatter_plots(raw_data, residual_df, lag,
                               manual_pacf_values[lag - 1],
                               acf_values[lag - 1])
    st.pyplot(fig)

# 显示对比表格
st.subheader("PACF 值对比")
styled_table = create_comparison_table(ar_coefs=(phi1, phi2, phi3),
                                       sample_data=ar3_data,
                                       manual_pacf=manual_pacf_values)
st.dataframe(styled_table, use_container_width=True)

# 使用说明
st.markdown("""
### 使用说明
1. 修改左侧边栏以调整模型参数
2. 每个滞后阶数都会生成两个图：
   - 原始数据的AFC散点图
   - 残差数据的PACF散点图
3. 拟合直线显示了两个变量之间的趋势关系
""")

import math
import random
import streamlit as st

st.title("연평균 기온 선형회귀 분석")

# 1. 데이터 생성 (1906 ~ 2025)
random.seed(42)
years = list(range(1906, 2026))
temps = []
for y in years:
    t = 10.8 + 0.012 * (y - 1906) + 0.00012 * ((y - 1906) ** 2) + random.gauss(0, 0.45)
    temps.append(t)

# 데이터 분할
def get_split(start_yr, end_yr):
    x, y = [], []
    for yr, tp in zip(years, temps):
        if start_yr <= yr <= end_yr:
            x.append(yr)
            y.append(tp)
    return x, y

x_test, y_test = get_split(2006, 2025)

# 선형회귀 학습 및 평가 함수 (최소제곱법)
def run_regression(x_train, y_train, x_eval, y_eval):
    n = len(x_train)
    mean_x = sum(x_train) / n
    mean_y = sum(y_train) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_train, y_train))
    den = sum((x - mean_x) ** 2 for x in x_train)

    slope = num / den
    intercept = mean_y - (slope * mean_x)

    # 예측
    y_pred = [slope * x + intercept for x in x_eval]

    # 평가지표
    m = len(y_eval)
    mae = sum(abs(act - prd) for act, prd in zip(y_eval, y_pred)) / m
    mse = sum((act - prd) ** 2 for act, prd in zip(y_eval, y_pred)) / m

    mean_test_y = sum(y_eval) / m
    ss_tot = sum((act - mean_test_y) ** 2 for act in y_eval)
    ss_res = sum((act - prd) ** 2 for act, prd in zip(y_eval, y_pred))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return slope, intercept, mae, mse, r2

# 모델 실행
x_all, y_all = get_split(1906, 2025)
x_50, y_50 = get_split(1956, 2005)
x_100, y_100 = get_split(1906, 2005)

s_all, i_all, mae_all, mse_all, r2_all = run_regression(
    x_all, y_all, x_test, y_test
)
s_50, i_50, mae_50, mse_50, r2_50 = run_regression(
    x_50, y_50, x_test, y_test
)
s_100, i_100, mae_100, mse_100, r2_100 = run_regression(
    x_100, y_100, x_test, y_test
)

# 결과 출력
st.subheader("모델별 테스트 데이터(2006~2025) 예측 성능 비교")

data = {
    "모델": [
        "전체 데이터 (1906~2025)",
        "최근 50년 학습 (1956~2005)",
        "최근 100년 학습 (1906~2005)",
    ],
    "기울기 (Slope)": [f"{s_all:.4f}", f"{s_50:.4f}", f"{s_100:.4f}"],
    "MAE": [f"{mae_all:.4f}", f"{mae_50:.4f}", f"{mae_100:.4f}"],
    "MSE": [f"{mse_all:.4f}", f"{mse_50:.4f}", f"{mse_100:.4f}"],
    "R²": [f"{r2_all:.4f}", f"{r2_50:.4f}", f"{r2_100:.4f}"],
}

st.table(data)

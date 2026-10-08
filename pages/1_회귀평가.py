import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="서울 연평균 기온 선형회귀 비교",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 비교")
st.write(
    "과거 기온 데이터를 이용해 선형회귀 모델을 학습하고, "
    "최근 20년(2006~2025년)의 기온을 얼마나 잘 예측하는지 비교합니다."
)

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(URL, encoding="cp949")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온
    yearly = (
        df.dropna(subset=["평균기온"])
        .groupby("연도")["평균기온"]
        .agg(["mean", "count"])
        .reset_index()
    )

    yearly.columns = ["연도", "평균기온", "관측일수"]

    # 1년 관측일수가 300일 이상인 연도만 사용
    yearly = yearly[yearly["관측일수"] >= 300]

    return yearly.sort_values("연도").reset_index(drop=True)


try:
    yearly = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.write(e)
    st.stop()


# --------------------------------------------------
# 분석 대상 데이터
# --------------------------------------------------
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# --------------------------------------------------
# 데이터 존재 여부 확인
# --------------------------------------------------
if len(test) == 0:
    st.error("2006~2025년 테스트 데이터가 없습니다.")
    st.stop()

if len(train_50) == 0 or len(train_100) == 0:
    st.error("훈련 데이터가 충분하지 않습니다.")
    st.stop()


# --------------------------------------------------
# 선형회귀 모델
# --------------------------------------------------
def make_model(train_data):

    X = train_data[["연도"]]
    y = train_data["평균기온"]

    model = LinearRegression()
    model.fit(X, y)

    return model


model_50 = make_model(train_50)
model_100 = make_model(train_100)


# --------------------------------------------------
# 테스트 데이터 예측
# --------------------------------------------------
X_test = test[["연도"]]
y_test = test["평균기온"]

pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)


# --------------------------------------------------
# 평가 지표
# --------------------------------------------------
mae_50 = mean_absolute_error(y_test, pred_50)
mse_50 = mean_squared_error(y_test, pred_50)
r2_50 = r2_score(y_test, pred_50)

mae_100 = mean_absolute_error(y_test, pred_100)
mse_100 = mean_squared_error(y_test, pred_100)
r2_100 = r2_score(y_test, pred_100)


# --------------------------------------------------
# 회귀선 정보
# --------------------------------------------------
slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_


# --------------------------------------------------
# 데이터 범위 표시
# --------------------------------------------------
st.subheader("📌 데이터 분할")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "50년 훈련 데이터",
        f"{train_50['연도'].min()}~{train_50['연도'].max()}",
        f"{len(train_50)}개 연도"
    )

with col2:
    st.metric(
        "100년 훈련 데이터",
        f"{train_100['연도'].min()}~{train_100['연도'].max()}",
        f"{len(train_100)}개 연도"
    )

with col3:
    st.metric(
        "공통 테스트 데이터",
        f"{test['연도'].min()}~{test['연도'].max()}",
        f"{len(test)}개 연도"
    )


st.info(
    "두 모델 모두 2006~2025년 데이터를 학습에 사용하지 않았습니다. "
    "따라서 같은 테스트 데이터를 이용해 두 모델의 미래 예측 성능을 공정하게 비교합니다."
)


# --------------------------------------------------
# 회귀선 비교
# --------------------------------------------------
st.subheader("📈 50년 학습 vs 100년 학습 회귀선")

fig = go.Figure()

# 실제 연평균 기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(size=6)
    )
)

# 회귀선용 연도
line_years = np.arange(
    min(train_100["연도"].min(), train_50["연도"].min()),
    2026
).reshape(-1, 1)

# 100년 회귀선
fig.add_trace(
    go.Scatter(
        x=line_years.flatten(),
        y=model_100.predict(line_years),
        mode="lines",
        name="1906~2005 학습 회귀선"
    )
)

# 50년 회귀선
fig.add_trace(
    go.Scatter(
        x=line_years.flatten(),
        y=model_50.predict(line_years),
        mode="lines",
        name="1956~2005 학습 회귀선"
    )
)

# 테스트 구간 시작선
fig.add_vline(
    x=2006,
    line_dash="dash",
    annotation_text="테스트 시작"
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=550
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 기울기 비교
# --------------------------------------------------
st.subheader("📐 회귀선의 기울기 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "최근 50년 회귀선 기울기",
        f"{slope_50:.4f} ℃/년"
    )

with col2:
    st.metric(
        "최근 100년 회귀선 기울기",
        f"{slope_100:.4f} ℃/년"
    )

st.write(
    f"50년 학습 모델의 기울기는 **{slope_50:.4f}℃/년**, "
    f"100년 학습 모델의 기울기는 **{slope_100:.4f}℃/년**입니다."
)

if slope_50 > slope_100:
    st.info(
        "최근 50년 데이터를 이용한 회귀선의 기울기가 더 큽니다. "
        "이는 장기간 전체 추세보다 최근 수십 년의 기온 상승 추세가 더 가파르게 나타났다는 것을 의미합니다."
    )
elif slope_50 < slope_100:
    st.info(
        "최근 100년 데이터를 이용한 회귀선의 기울기가 더 큽니다. "
        "장기간의 기온 변화까지 포함하면 전체적인 상승 추세가 더 크게 나타납니다."
    )
else:
    st.info("두 모델의 기울기가 거의 같습니다.")


# --------------------------------------------------
# 예측 성능 비교
# --------------------------------------------------
st.subheader("🎯 최근 20년 테스트 데이터 예측 성능")

result = pd.DataFrame({
    "모델": [
        "최근 50년 학습 (1956~2005)",
        "최근 100년 학습 (1906~2005)"
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE (℃²)": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    result.style.format({
        "MAE (℃)": "{:.3f}",
        "MSE (℃²)": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True
)


# --------------------------------------------------
# 성능 설명
# --------------------------------------------------
st.markdown("""
### 지표 해석

- **MAE**: 실제 기온과 예측 기온의 평균적인 차이입니다. 작을수록 좋습니다.
- **MSE**: 예측 오차를 제곱하여 평균한 값입니다. 작을수록 좋으며 큰 오차에 더 민감합니다.
- **R²**: 실제 기온의 변화를 회귀모델이 얼마나 설명하는지를 나타냅니다. 1에 가까울수록 좋습니다.
""")


# --------------------------------------------------
# 실제값 vs 예측값
# --------------------------------------------------
st.subheader("🔎 2006~2025년 실제 기온과 예측 기온")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="lines+markers",
        name="실제 기온"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines+markers",
        name="50년 학습 예측"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines+markers",
        name="100년 학습 예측"
    )
)

fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# 연도별 예측값 표
# --------------------------------------------------
st.subheader("📋 테스트 데이터 예측 결과")

prediction_table = test[["연도", "평균기온"]].copy()

prediction_table["50년 학습 예측"] = pred_50
prediction_table["100년 학습 예측"] = pred_100

prediction_table["50년 오차"] = (
    prediction_table["평균기온"] -
    prediction_table["50년 학습 예측"]
)

prediction_table["100년 오차"] = (
    prediction_table["평균기온"] -
    prediction_table["100년 학습 예측"]
)

st.dataframe(
    prediction_table.style.format({
        "평균기온": "{:.2f}",
        "50년 학습 예측": "{:.2f}",
        "100년 학습 예측": "{:.2f}",
        "50년 오차": "{:.2f}",
        "100년 오차": "{:.2f}"
    }),
    use_container_width=True
)


# --------------------------------------------------
# 최종 결론 자동 생성
# --------------------------------------------------
st.subheader("📝 분석 결과")

if mae_50 < mae_100:
    better_mae = "최근 50년 학습 모델"
else:
    better_mae = "최근 100년 학습 모델"

if mse_50 < mse_100:
    better_mse = "최근 50년 학습 모델"
else:
    better_mse = "최근 100년 학습 모델"

if r2_50 > r2_100:
    better_r2 = "최근 50년 학습 모델"
else:
    better_r2 = "최근 100년 학습 모델"

st.write(
    f"""
    **① 기울기:** 최근 50년 학습 모델의 기울기는 
    **{slope_50:.4f}℃/년**, 최근 100년 학습 모델은 
    **{slope_100:.4f}℃/년**으로 나타났습니다.

    **② MAE:** {better_mae}이 더 낮은 MAE를 보여 평균적인 예측 오차가 더 작았습니다.

    **③ MSE:** {better_mse}이 더 낮은 MSE를 보여 큰 예측 오차를 줄이는 데 더 유리했습니다.

    **④ R²:** {better_r2}이 더 높은 R²를 보여 테스트 기간의 기온 변화를 더 잘 설명했습니다.

    따라서 단순히 훈련 데이터가 많다고 해서 항상 최근 기온을 더 잘 예측하는 것은 아니며,
    **어떤 기간의 데이터를 학습에 포함하느냐에 따라 회귀선의 기울기와 예측 성능이 달라질 수 있습니다.**
    """
)

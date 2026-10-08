import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="연평균 기온 회귀 평가",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 연평균 기온 선형회귀 모델 평가")

st.write(
    "과거 서울 연평균 기온 데이터를 이용하여 "
    "최근 50년과 최근 100년의 선형회귀 모델을 만들고, "
    "공통 테스트 데이터인 2006~2025년의 기온을 얼마나 잘 예측하는지 비교합니다."
)


# =========================================================
# 데이터 주소
# =========================================================
URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/"
    "data/seoul.csv"
)


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():

    # GitHub CSV는 UTF-8로 읽기
    df = pd.read_csv(URL, encoding="utf-8")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 관측일수 계산
    yearly = (
        df.dropna(subset=["평균기온"])
        .groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 1년 관측일수가 300일 이상인 연도만 사용
    yearly = yearly[
        yearly["관측일수"] >= 300
    ]

    # 연도순 정렬
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


# =========================================================
# 데이터 불러오기
# =========================================================
try:
    yearly = load_data()

except Exception as e:

    st.error("데이터를 불러오는 중 오류가 발생했습니다.")

    st.code(str(e))

    st.stop()


# =========================================================
# 데이터 분할
# =========================================================

# 최근 50년 학습
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()


# 최근 100년 학습
train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()


# 공통 테스트 데이터
test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# =========================================================
# 데이터 확인
# =========================================================

if len(train_50) == 0:
    st.error("1956~2005년 훈련 데이터가 없습니다.")
    st.stop()

if len(train_100) == 0:
    st.error("1906~2005년 훈련 데이터가 없습니다.")
    st.stop()

if len(test) == 0:
    st.error("2006~2025년 테스트 데이터가 없습니다.")
    st.stop()


# =========================================================
# 데이터 분할 표시
# =========================================================
st.subheader("📊 데이터 분할")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 학습",
        "1956~2005",
        f"{len(train_50)}개 연도"
    )

with col2:
    st.metric(
        "최근 100년 학습",
        "1906~2005",
        f"{len(train_100)}개 연도"
    )

with col3:
    st.metric(
        "공통 테스트",
        "2006~2025",
        f"{len(test)}개 연도"
    )


st.info(
    "두 모델 모두 2006~2025년 데이터를 학습에 사용하지 않았습니다. "
    "따라서 동일한 테스트 데이터를 이용해 두 모델의 예측 성능을 비교합니다."
)


# =========================================================
# 선형회귀 함수
# =========================================================
def make_model(data):

    X = data[["연도"]]
    y = data["평균기온"]

    model = LinearRegression()

    model.fit(X, y)

    return model


# =========================================================
# 모델 생성
# =========================================================
model_50 = make_model(train_50)
model_100 = make_model(train_100)


# =========================================================
# 테스트 데이터 예측
# =========================================================
X_test = test[["연도"]]

y_test = test["평균기온"]


pred_50 = model_50.predict(X_test)

pred_100 = model_100.predict(X_test)


# =========================================================
# 회귀선 정보
# =========================================================
slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_


# =========================================================
# 회귀식 표시
# =========================================================
st.subheader("📐 만들어진 회귀식")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 최근 50년 학습")

    st.latex(
        f"y = {slope_50:.5f}x + {intercept_50:.2f}"
    )

    st.write(
        f"기울기: **{slope_50:.5f} ℃/년**"
    )


with col2:

    st.markdown("### 최근 100년 학습")

    st.latex(
        f"y = {slope_100:.5f}x + {intercept_100:.2f}"
    )

    st.write(
        f"기울기: **{slope_100:.5f} ℃/년**"
    )


# =========================================================
# 회귀선 비교 그래프
# =========================================================
st.subheader("📈 50년 학습과 100년 학습 회귀선 비교")


fig = go.Figure()


# 실제 연평균 기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(size=5)
    )
)


# 회귀선을 그릴 연도
line_years = np.arange(
    yearly["연도"].min(),
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


# 테스트 시작점 표시
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


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 기울기 비교
# =========================================================
st.subheader("📐 회귀선 기울기 비교")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "최근 50년 기울기",
        f"{slope_50:.5f} ℃/년"
    )


with col2:

    st.metric(
        "최근 100년 기울기",
        f"{slope_100:.5f} ℃/년"
    )


difference = slope_50 - slope_100


st.write(
    f"두 회귀선의 기울기 차이는 "
    f"**{difference:.5f} ℃/년**입니다."
)


if slope_50 > slope_100:

    st.info(
        "최근 50년을 학습한 회귀선의 기울기가 더 큽니다. "
        "따라서 최근 수십 년의 기온 상승 추세가 "
        "1906년부터의 장기 추세보다 더 가파르게 나타납니다."
    )

elif slope_50 < slope_100:

    st.info(
        "최근 100년을 학습한 회귀선의 기울기가 더 큽니다. "
        "장기간의 기온 변화까지 포함했을 때 "
        "전체적인 상승 추세가 더 크게 나타납니다."
    )

else:

    st.info(
        "두 모델의 기울기가 거의 같습니다."
    )


# =========================================================
# 평가 지표 계산
# =========================================================

# 50년 모델
mae_50 = mean_absolute_error(
    y_test,
    pred_50
)

mse_50 = mean_squared_error(
    y_test,
    pred_50
)

r2_50 = r2_score(
    y_test,
    pred_50
)


# 100년 모델
mae_100 = mean_absolute_error(
    y_test,
    pred_100
)

mse_100 = mean_squared_error(
    y_test,
    pred_100
)

r2_100 = r2_score(
    y_test,
    pred_100
)


# =========================================================
# 평가 결과
# =========================================================
st.subheader("🎯 2006~2025년 테스트 데이터 예측 성능")


result = pd.DataFrame({

    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],

    "훈련 기간": [
        "1956~2005",
        "1906~2005"
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
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 평가 지표 설명
# =========================================================
st.markdown("""
### 📌 평가 지표의 의미

**MAE (평균 절대 오차)**  
실제 기온과 예측 기온의 차이를 절댓값으로 계산한 평균입니다.  
→ **작을수록 좋습니다.**

**MSE (평균 제곱 오차)**  
예측 오차를 제곱한 뒤 평균한 값입니다.  
→ **작을수록 좋으며 큰 오차에 더 민감합니다.**

**R² (결정계수)**  
회귀모델이 실제 기온의 변화를 얼마나 설명하는지를 나타냅니다.  
→ **1에 가까울수록 좋습니다.**
""")


# =========================================================
# 실제값 vs 예측값
# =========================================================
st.subheader("🔎 2006~2025년 실제 기온과 예측 기온")


fig2 = go.Figure()


# 실제값
fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="lines+markers",
        name="실제 기온"
    )
)


# 50년 모델 예측
fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines+markers",
        name="50년 학습 예측"
    )
)


# 100년 모델 예측
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


st.plotly_chart(
    fig2,
    use_container_width=True
)


# =========================================================
# 테스트 데이터 예측표
# =========================================================
st.subheader("📋 연도별 예측 결과")


prediction_table = test[
    ["연도", "평균기온"]
].copy()


prediction_table["50년 학습 예측"] = pred_50

prediction_table["100년 학습 예측"] = pred_100


prediction_table["50년 오차"] = (
    prediction_table["평균기온"]
    - prediction_table["50년 학습 예측"]
)


prediction_table["100년 오차"] = (
    prediction_table["평균기온"]
    - prediction_table["100년 학습 예측"]
)


st.dataframe(
    prediction_table.style.format({
        "평균기온": "{:.2f}",
        "50년 학습 예측": "{:.2f}",
        "100년 학습 예측": "{:.2f}",
        "50년 오차": "{:.2f}",
        "100년 오차": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 어느 모델이 더 좋은지 비교
# =========================================================
st.subheader("🏆 두 모델 비교")


if mae_50 < mae_100:
    mae_result = "50년 학습 모델"
else:
    mae_result = "100년 학습 모델"


if mse_50 < mse_100:
    mse_result = "50년 학습 모델"
else:
    mse_result = "100년 학습 모델"


if r2_50 > r2_100:
    r2_result = "50년 학습 모델"
else:
    r2_result = "100년 학습 모델"


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "MAE가 더 좋은 모델",
        mae_result
    )


with col2:

    st.metric(
        "MSE가 더 좋은 모델",
        mse_result
    )


with col3:

    st.metric(
        "R²가 더 좋은 모델",
        r2_result
    )


# =========================================================
# 최종 분석
# =========================================================
st.subheader("📝 분석 결과")


st.write(
    f"""
이번 분석에서는 **2006~2025년을 공통 테스트 데이터**로 설정하고,
1956~2005년의 최근 50년 데이터와 1906~2005년의 최근 100년 데이터를
각각 학습시켜 선형회귀 모델을 비교했습니다.

최근 50년 학습 모델의 회귀선 기울기는
**{slope_50:.5f} ℃/년**이고,
최근 100년 학습 모델의 기울기는
**{slope_100:.5f} ℃/년**입니다.

테스트 데이터에 대한 MAE는 각각
**{mae_50:.3f}℃**, **{mae_100:.3f}℃**,
MSE는 각각
**{mse_50:.3f}**, **{mse_100:.3f}**,
R²는 각각
**{r2_50:.3f}**, **{r2_100:.3f}**로 나타났습니다.

따라서 단순히 더 많은 과거 데이터를 사용하는 것이 항상
최근 기온을 더 정확하게 예측하는 것은 아니며,
**어떤 기간을 학습 데이터로 선택하느냐에 따라 회귀선의 기울기와
예측 성능이 달라질 수 있다는 점을 확인할 수 있습니다.**
"""
)


# =========================================================
# 데이터 확인용
# =========================================================
with st.expander("📂 사용된 연도별 데이터 확인"):

    st.dataframe(
        yearly,
        use_container_width=True,
        hide_index=True
    )

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error


st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균기온 데이터를 이용하여 "
    "기온의 변화 추세를 분석하고 미래의 기온을 예측합니다."
)


# =========================================================
# 데이터
# =========================================================

URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/"
    "data/seoul.csv"
)


@st.cache_data
def load_data():

    df = pd.read_csv(
        URL,
        encoding="utf-8-sig"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df["연도"] = df["날짜"].dt.year

    # 2025년 이후 제외
    df = df[df["연도"] <= 2025]

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

    # 관측일수가 300일 미만인 연도 제외
    yearly = yearly[
        yearly["관측일수"] >= 300
    ]

    yearly = yearly.sort_values(
        "연도"
    ).reset_index(drop=True)

    return yearly


try:
    yearly = load_data()

except Exception as e:

    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()


# =========================================================
# 기본 정보
# =========================================================

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
year_count = len(yearly)

st.subheader("📊 사용한 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "사용한 연도 수",
        f"{year_count}개"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{start_year}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{end_year}년"
    )

st.info(
    "2025년까지의 데이터 중 관측일수가 300일 이상인 "
    "연도만 분석에 사용했습니다."
)


# =========================================================
# 계산용 연도
# =========================================================
# 실제 연도는 그래프에 그대로 표시하고,
# 계산할 때만 2000을 빼서 숫자를 작게 만듦

yearly["계산연도"] = yearly["연도"] - 2000


# =========================================================
# 1. 전체 데이터 선형회귀
# =========================================================

st.divider()

st.header("📈 연평균기온과 선형회귀")

X = yearly[["계산연도"]]
y = yearly["평균기온"]

linear_model = LinearRegression()

linear_model.fit(X, y)

yearly["선형회귀예측"] = linear_model.predict(X)

correlation = (
    yearly["계산연도"]
    .corr(yearly["평균기온"])
)

slope = linear_model.coef_[0]
intercept = linear_model.intercept_


# 상관계수

st.subheader("🔗 상관계수")

st.metric(
    "연도와 평균기온의 상관계수",
    f"{correlation:.3f}"
)

if correlation > 0:

    st.write(
        "상관계수가 양수이므로 시간이 지날수록 "
        "연평균기온이 높아지는 경향이 있습니다."
    )

elif correlation < 0:

    st.write(
        "상관계수가 음수이므로 시간이 지날수록 "
        "연평균기온이 낮아지는 경향이 있습니다."
    )

else:

    st.write(
        "연도와 평균기온 사이에 뚜렷한 선형 관계가 나타나지 않습니다."
    )


# 회귀식

st.subheader("📐 선형회귀식")

st.latex(
    f"y = {slope:.4f}x + {intercept:.4f}"
)

st.write(
    "x는 2000년을 기준으로 변환한 계산용 연도이고, "
    "y는 예상 연평균기온(℃)입니다."
)

st.write(
    f"회귀선의 기울기: **{slope:.4f} ℃/년**"
)


# =========================================================
# 선형회귀 그래프
# =========================================================

fig_linear = go.Figure()

fig_linear.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6),
        hovertemplate=
            "연도: %{x}<br>"
            "평균기온: %{y:.2f}℃"
            "<extra></extra>"
    )
)

fig_linear.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["선형회귀예측"],
        mode="lines",
        name="선형회귀선",
        hovertemplate=
            "연도: %{x}<br>"
            "예상기온: %{y:.2f}℃"
            "<extra></extra>"
    )
)

fig_linear.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=600
)

st.plotly_chart(
    fig_linear,
    use_container_width=True
)


# =========================================================
# 연도별 예상 기온
# =========================================================

st.subheader("🔮 연도별 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

selected_x = pd.DataFrame({
    "계산연도": [
        selected_year - 2000
    ]
})

selected_prediction = linear_model.predict(
    selected_x
)[0]

st.metric(
    f"{selected_year}년 선형회귀 예상 연평균기온",
    f"{selected_prediction:.2f} ℃"
)

actual = yearly[
    yearly["연도"] == selected_year
]

if len(actual) > 0:

    actual_temp = actual["평균기온"].iloc[0]

    difference = (
        selected_prediction - actual_temp
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "실제 연평균기온",
            f"{actual_temp:.2f} ℃"
        )

    with col2:
        st.metric(
            "회귀선과 실제값의 차이",
            f"{difference:+.2f} ℃"
        )


# =========================================================
# 2. 1차 / 3차 / 9차 곡선 비교
# =========================================================

st.divider()

st.header("📈 1차·3차·9차 곡선을 이용한 기온 예측")

st.write(
    "2005년 이전의 데이터를 훈련 데이터로 사용하고, "
    "2005년부터의 데이터를 테스트 데이터로 사용하여 "
    "1차·3차·9차 곡선의 예측 성능을 비교합니다."
)


# =========================================================
# 훈련 / 테스트 분리
# =========================================================

# 2005년 이전 = 훈련
train = yearly[
    yearly["연도"] < 2005
].copy()

# 2005년부터 = 테스트
test = yearly[
    yearly["연도"] >= 2005
].copy()


# =========================================================
# 훈련 / 테스트 데이터 정보
# =========================================================

st.subheader("📊 훈련 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "훈련용 연도",
        f"{len(train)}개"
    )

    st.write(
        f"{int(train['연도'].min())}년 ~ "
        f"{int(train['연도'].max())}년"
    )

with col2:

    st.metric(
        "테스트용 연도",
        f"{len(test)}개"
    )

    st.write(
        f"{int(test['연도'].min())}년 ~ "
        f"{int(test['연도'].max())}년"
    )


st.info(
    "2005년 이전 데이터만 모델을 학습하는 데 사용했습니다. "
    "2005년 이후 데이터는 학습에 사용하지 않고 "
    "모델의 성능을 평가하는 데만 사용합니다."
)


# =========================================================
# X, y
# =========================================================

X_train = train[["계산연도"]]
y_train = train["평균기온"]

X_test = test[["계산연도"]]
y_test = test["평균기온"]


# =========================================================
# 모델
# =========================================================

models = {

    "1차": make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=1),
        LinearRegression()
    ),

    "3차": make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3),
        LinearRegression()
    ),

    "9차": make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=9),
        LinearRegression()
    )
}


# =========================================================
# 학습 및 테스트
# =========================================================

results = []
predictions = {}

for name, model in models.items():

    # 훈련 데이터로만 학습
    model.fit(
        X_train,
        y_train
    )

    # 테스트 데이터 예측
    test_prediction = model.predict(
        X_test
    )

    predictions[name] = test_prediction

    # 테스트 데이터 MAE
    mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    # 2050년 예측
    x_2050 = pd.DataFrame({
        "계산연도": [2050 - 2000]
    })

    prediction_2050 = model.predict(
        x_2050
    )[0]

    results.append({
        "모델": name,
        "테스트 평균 오차(℃)": mae,
        "2050년 예측값(℃)": prediction_2050
    })


# =========================================================
# 결과표
# =========================================================

st.subheader("🎯 테스트 성능과 2050년 예측")

result_df = pd.DataFrame(results)

display_df = result_df.copy()

display_df["테스트 평균 오차(℃)"] = (
    display_df["테스트 평균 오차(℃)"]
    .map(lambda x: f"{x:.2f} ℃")
)

display_df["2050년 예측값(℃)"] = (
    display_df["2050년 예측값(℃)"]
    .map(lambda x: f"{x:.2f} ℃")
)

st.table(display_df)

st.caption(
    "MAE가 작을수록 테스트 데이터의 실제 기온을 "
    "평균적으로 더 가깝게 예측한 것입니다."
)


# =========================================================
# 가장 좋은 모델
# =========================================================

best_index = result_df[
    "테스트 평균 오차(℃)"
].idxmin()

best_model_name = result_df.loc[
    best_index,
    "모델"
]

best_mae = result_df.loc[
    best_index,
    "테스트 평균 오차(℃)"
]

st.success(
    f"테스트 데이터에서 평균 오차가 가장 작은 모델은 "
    f"**{best_model_name}**이며, "
    f"평균적으로 **{best_mae:.2f}℃** 차이가 났습니다."
)


# =========================================================
# 훈련 + 테스트 + 곡선 그래프
# =========================================================

st.subheader("📊 훈련 데이터와 테스트 데이터 및 곡선")

fig_curve = go.Figure()


# 훈련 데이터

fig_curve.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["평균기온"],
        mode="markers",
        name="훈련 데이터",
        marker=dict(size=6),
        hovertemplate=
            "연도: %{x}<br>"
            "평균기온: %{y:.2f}℃"
            "<extra></extra>"
    )
)


# 테스트 데이터

fig_curve.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="markers",
        name="테스트 데이터",
        marker=dict(
            size=7,
            symbol="circle-open"
        ),
        hovertemplate=
            "연도: %{x}<br>"
            "실제기온: %{y:.2f}℃"
            "<extra></extra>"
    )
)


# 곡선용 연도

curve_years = np.linspace(
    start_year,
    2050,
    500
)

curve_X = pd.DataFrame({
    "계산연도": curve_years - 2000
})


# 1차 / 3차 / 9차 곡선

for name, model in models.items():

    curve_prediction = model.predict(
        curve_X
    )

    fig_curve.add_trace(
        go.Scatter(
            x=curve_years,
            y=curve_prediction,
            mode="lines",
            name=f"{name} 곡선"
        )
    )


# 2005년 기준선

fig_curve.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="2005년: 학습 → 테스트"
)

fig_curve.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=650
)

st.plotly_chart(
    fig_curve,
    use_container_width=True
)


# =========================================================
# 2050년 예측 비교
# =========================================================

st.subheader("🔮 2050년 예측 비교")

cols = st.columns(3)

x_2050 = pd.DataFrame({
    "계산연도": [2050 - 2000]
})

for col, (name, model) in zip(
    cols,
    models.items()
):

    value = model.predict(
        x_2050
    )[0]

    with col:

        st.metric(
            f"{name} 곡선",
            f"{value:.2f} ℃"
        )


# =========================================================
# 테스트 데이터별 실제값 / 예측값
# =========================================================

with st.expander(
    "📋 테스트 데이터별 실제값과 예측값 보기"
):

    comparison = test[
        ["연도", "평균기온"]
    ].copy()

    comparison["1차 예측"] = predictions["1차"]
    comparison["3차 예측"] = predictions["3차"]
    comparison["9차 예측"] = predictions["9차"]

    comparison = comparison.round(2)

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 결과 해석
# =========================================================

st.subheader("💡 결과 해석")

st.write(
    """
**1차 곡선**은 직선이므로 전체적인 기온의 상승·하락 추세를 나타냅니다.

**3차 곡선**은 직선보다 유연하여 기온 변화의 굽은 추세를 표현할 수 있습니다.

**9차 곡선**은 복잡한 변화를 표현할 수 있지만,
훈련 데이터에 지나치게 맞춰지는 **과적합**이 발생할 가능성이 있습니다.

따라서 이번 탐구에서는 훈련 데이터에 얼마나 잘 맞는지가 아니라,
학습에 사용하지 않은 **테스트 데이터의 MAE가 얼마나 작은지**를 기준으로
세 모델의 예측 성능을 비교했습니다.

또한 고차 다항식을 계산할 때 큰 연도를 그대로 사용하면
거듭제곱 과정에서 계산이 불안정해질 수 있으므로,
연도에서 2000을 빼고 표준화하여 계산했습니다.
"""
)

# =========================================================
# 데이터 보기
# =========================================================

with st.expander(
    "📋 사용한 연평균기온 데이터 보기"
):

    st.dataframe(
        yearly[
            [
                "연도",
                "관측일수",
                "평균기온"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

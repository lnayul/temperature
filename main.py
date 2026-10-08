import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "연평균기온을 이용해 1차, 3차, 9차 곡선을 학습하고 "
    "학습에 사용하지 않은 테스트 데이터로 성능을 비교합니다."
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

    df = pd.read_csv(
        URL,
        encoding="utf-8-sig"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년 이후 데이터 제외
    df = df[df["연도"] <= 2025]

    # 연도별 평균기온과 관측일수
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

    return yearly.sort_values(
        "연도"
    ).reset_index(drop=True)


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
# 연도를 작은 숫자로 변환
# =========================================================
# 1908년을 기준으로 몇 년이 지났는지를 사용
yearly["경과연수"] = yearly["연도"] - 1908


# =========================================================
# 훈련 / 테스트 분리
# =========================================================
# 2005년 이전 → 훈련
# 2005년부터 → 테스트

train = yearly[
    yearly["연도"] < 2005
].copy()

test = yearly[
    yearly["연도"] >= 2005
].copy()


# 데이터가 제대로 나뉘었는지 확인
if len(train) == 0 or len(test) == 0:

    st.error(
        "훈련 데이터 또는 테스트 데이터가 없습니다."
    )

    st.stop()


# =========================================================
# 데이터 개수
# =========================================================
st.subheader("📊 훈련 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "훈련용 연도",
        f"{len(train)}개"
    )

    st.write(
        f"**{int(train['연도'].min())}년 ~ "
        f"{int(train['연도'].max())}년**"
    )

with col2:

    st.metric(
        "테스트용 연도",
        f"{len(test)}개"
    )

    st.write(
        f"**{int(test['연도'].min())}년 ~ "
        f"{int(test['연도'].max())}년**"
    )


st.info(
    "2005년 이전 데이터만 학습에 사용하고, "
    "2005년부터 2025년까지의 데이터는 학습에 사용하지 않은 "
    "테스트 데이터로 성능을 평가합니다."
)


# =========================================================
# X, y
# =========================================================
X_train = train[["경과연수"]]
y_train = train["평균기온"]

X_test = test[["경과연수"]]
y_test = test["평균기온"]


# =========================================================
# 1차 / 3차 / 9차 모델
# =========================================================
models = {

    "1차": make_pipeline(
        PolynomialFeatures(degree=1),
        LinearRegression()
    ),

    "3차": make_pipeline(
        PolynomialFeatures(degree=3),
        LinearRegression()
    ),

    "9차": make_pipeline(
        PolynomialFeatures(degree=9),
        LinearRegression()
    )
}


# =========================================================
# 학습 + 테스트 평가
# =========================================================
results = []
predictions = {}


for name, model in models.items():

    # 반드시 훈련 데이터만 사용
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
        "경과연수": [2050 - 1908]
    })

    prediction_2050 = model.predict(
        x_2050
    )[0]

    results.append({
        "모델": name,
        "테스트 평균 오차(°C)": mae,
        "2050년 예측(°C)": prediction_2050
    })


# =========================================================
# 결과 표
# =========================================================
st.subheader("🎯 곡선별 테스트 성능과 2050년 예측")

result_df = pd.DataFrame(results)

display_df = result_df.copy()

display_df["테스트 평균 오차(°C)"] = (
    display_df["테스트 평균 오차(°C)"]
    .map(lambda x: f"{x:.2f} °C")
)

display_df["2050년 예측(°C)"] = (
    display_df["2050년 예측(°C)"]
    .map(lambda x: f"{x:.2f} °C")
)

st.table(display_df)


st.caption(
    "MAE는 테스트 데이터에서 실제 연평균기온과 예측기온의 차이를 "
    "절댓값으로 계산한 평균입니다. 작을수록 좋습니다."
)


# =========================================================
# 가장 좋은 모델
# =========================================================
best_index = result_df[
    "테스트 평균 오차(°C)"
].idxmin()

best_model_name = result_df.loc[
    best_index,
    "모델"
]

best_mae = result_df.loc[
    best_index,
    "테스트 평균 오차(°C)"
]


st.success(
    f"테스트 데이터에서 가장 작은 평균 오차를 보인 모델은 "
    f"**{best_model_name}**이며, 평균적으로 "
    f"**{best_mae:.2f}°C** 차이가 났습니다."
)


# =========================================================
# 그래프
# =========================================================
st.subheader("📈 훈련 데이터와 테스트 데이터 + 곡선")

fig = go.Figure()


# 훈련 데이터
fig.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["평균기온"],
        mode="markers",
        name="훈련 데이터",
        marker=dict(
            size=6
        ),
        hovertemplate=
            "연도: %{x}<br>"
            "평균기온: %{y:.2f}°C"
            "<extra></extra>"
    )
)


# 테스트 데이터
fig.add_trace(
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
            "평균기온: %{y:.2f}°C"
            "<extra></extra>"
    )
)


# 곡선을 그릴 연도
curve_years = np.linspace(
    int(yearly["연도"].min()),
    2050,
    500
)

curve_X = pd.DataFrame({
    "경과연수": curve_years - 1908
})


# 1차 / 3차 / 9차 곡선
for name, model in models.items():

    curve_prediction = model.predict(
        curve_X
    )

    fig.add_trace(
        go.Scatter(
            x=curve_years,
            y=curve_prediction,
            mode="lines",
            name=f"{name} 곡선"
        )
    )


# 2005년 경계선
fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="2005년: 학습 → 테스트"
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (°C)",
    hovermode="x unified",
    height=650
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 2050년 예측 비교
# =========================================================
st.subheader("🔮 2050년 예측 비교")

cols = st.columns(3)

x_2050 = pd.DataFrame({
    "경과연수": [2050 - 1908]
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
            f"{value:.2f} °C"
        )


# =========================================================
# 테스트 데이터별 예측
# =========================================================
with st.expander("📋 테스트 데이터별 실제값과 예측값"):

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
# 해석
# =========================================================
st.subheader("💡 해석")

st.write(
    """
**1차 곡선**은 직선이기 때문에 전체적인 상승 또는 하락 추세를 나타냅니다.

**3차 곡선**은 직선보다 유연하게 기온의 굽은 추세를 표현할 수 있습니다.

**9차 곡선**은 훈련 데이터의 복잡한 변화를 더 자세하게 따라갈 수 있지만,
훈련 데이터에 지나치게 맞춰지는 과적합이 발생할 가능성이 있습니다.

따라서 이번 비교에서는 훈련 데이터에 얼마나 잘 맞는지가 아니라,
**학습에 사용하지 않은 2005년 이후 테스트 데이터에서 MAE가 얼마나 작은지**를 기준으로 모델을 평가합니다.
"""
)

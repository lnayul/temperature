import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


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
    "서울의 연도별 평균기온을 이용하여 선형회귀 모델을 만들고 "
    "연도에 따른 기온 변화를 살펴봅니다."
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
# 데이터 불러오기 및 전처리
# =========================================================
@st.cache_data
def load_data():

    # UTF-8로 읽기
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

    # 연도순 정렬
    yearly = yearly.sort_values(
        "연도"
    ).reset_index(drop=True)

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
# 회귀용 데이터
# =========================================================

# 독립변수:
# 1908년부터 지난 연수
yearly["경과연수"] = yearly["연도"] - 1908

X = yearly[["경과연수"]]
y = yearly["평균기온"]


# =========================================================
# 선형회귀 모델
# =========================================================
model = LinearRegression()

model.fit(X, y)


# 회귀계수
slope = model.coef_[0]
intercept = model.intercept_


# 예측값
yearly["예측기온"] = model.predict(X)


# =========================================================
# 상관계수
# =========================================================
correlation = yearly["경과연수"].corr(
    yearly["평균기온"]
)


# =========================================================
# 기본 정보
# =========================================================
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)


st.subheader("📊 회귀선에 사용한 데이터")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "사용한 연도 수",
        f"{data_count}개"
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
    f"2025년까지의 데이터 중 관측일수가 300일 이상인 "
    f"{data_count}개 연도를 회귀선 계산에 사용했습니다."
)


# =========================================================
# 상관계수
# =========================================================
st.subheader("📈 연도와 평균기온의 상관관계")


st.metric(
    "상관계수",
    f"{correlation:.3f}"
)


if correlation > 0:

    st.write(
        "상관계수가 양수이므로 시간이 지날수록 "
        "연평균 기온이 높아지는 경향이 나타납니다."
    )

elif correlation < 0:

    st.write(
        "상관계수가 음수이므로 시간이 지날수록 "
        "연평균 기온이 낮아지는 경향이 나타납니다."
    )

else:

    st.write(
        "상관계수가 0에 가까워 뚜렷한 선형 관계가 나타나지 않습니다."
    )


# =========================================================
# 회귀식
# =========================================================
st.subheader("📐 선형회귀식")


st.latex(
    f"y = {slope:.4f}x + {intercept:.4f}"
)


st.write(
    "여기서 x는 **1908년부터 지난 연수**이고, "
    "y는 예상 연평균 기온(℃)입니다."
)


st.write(
    f"회귀선의 기울기는 **{slope:.4f} ℃/년**입니다."
)


# =========================================================
# 산점도 + 회귀선
# =========================================================
st.subheader("🌡️ 연도별 평균기온과 회귀선")


fig = go.Figure()


# 실제 연평균 기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(
            size=6
        ),
        hovertemplate=
            "연도: %{x}<br>"
            "평균기온: %{y:.2f}℃"
            "<extra></extra>"
    )
)


# 회귀선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["예측기온"],
        mode="lines",
        name="회귀선",
        hovertemplate=
            "연도: %{x}<br>"
            "예상기온: %{y:.2f}℃"
            "<extra></extra>"
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=600
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 연도 선택
# =========================================================
st.subheader("🔮 연도별 예상 기온")


selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 선택한 연도의 경과연수
selected_elapsed = selected_year - 1908


# 선택한 연도의 예측기온
predicted_temperature = model.predict(
    [[selected_elapsed]]
)[0]


# =========================================================
# 선택 연도 결과
# =========================================================
st.markdown(
    f"### {selected_year}년 예상 연평균 기온"
)

st.metric(
    "예상 기온",
    f"{predicted_temperature:.2f} ℃"
)


# 실제 데이터가 있는 경우 실제값도 표시
actual_row = yearly[
    yearly["연도"] == selected_year
]


if len(actual_row) > 0:

    actual_temperature = actual_row[
        "평균기온"
    ].iloc[0]

    difference = (
        predicted_temperature
        - actual_temperature
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "실제 연평균 기온",
            f"{actual_temperature:.2f} ℃"
        )

    with col2:

        st.metric(
            "회귀선과 실제값의 차이",
            f"{difference:+.2f} ℃"
        )


else:

    st.info(
        "선택한 연도는 실제 데이터가 없어 "
        "회귀모델을 이용한 예상 기온만 표시합니다."
    )


# =========================================================
# 선택한 연도를 그래프에서 표시
# =========================================================
prediction_years = np.array(
    [selected_year]
)

prediction_values = model.predict(
    prediction_years.reshape(-1, 1)
)


fig2 = go.Figure()


# 실제 데이터
fig2.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(
            size=5
        )
    )
)


# 회귀선
line_years = np.arange(
    1900,
    2101
)

line_elapsed = line_years - 1908

line_values = model.predict(
    line_elapsed.reshape(-1, 1)
)


fig2.add_trace(
    go.Scatter(
        x=line_years,
        y=line_values,
        mode="lines",
        name="회귀선"
    )
)


# 선택 연도
fig2.add_trace(
    go.Scatter(
        x=[selected_year],
        y=prediction_values,
        mode="markers",
        name=f"{selected_year}년 예상",
        marker=dict(
            size=14,
            symbol="star"
        )
    )
)


fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=550
)


st.plotly_chart(
    fig2,
    use_container_width=True
)

```python
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
    "2005년 이전의 연평균기온으로 곡선을 학습하고, "
    "학습에 사용하지 않은 2005년 이후의 데이터로 성능을 평가합니다."
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

    # 관측일수가 300일 미만인 해 제외
    yearly = yearly[
        yearly["관측일수"] >= 300
    ]

    return yearly.sort_values("연도").reset_index(drop=True)


# =========================================================
# 데이터 준비
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
# 훈련 / 테스트 데이터 분리
# =========================================================
# 2005년 이전 → 훈련용
# 2005년부터 → 테스트용

train = yearly[
    yearly["연도"] < 2005
].copy()

test = yearly[
    yearly["연도"] >= 2005
].copy()


# =========================================================
# 데이터 개수 표시
# =========================================================
st.subheader("📊 훈련 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "훈련용 연도",
        f"{len(train)}개"
    )
    st.write(
        f"범위: **{int(train['연도'].min())}년 ~ "
        f"{int(train['연도'].max())}년**"
    )

with col2:
    st.metric(
        "테스트용 연도",
        f"{len(test)}개"
    )
    st.write(
        f"범위: **{int(test['연도'].min())}년 ~ "
        f"{int(test['연도'].max())}년**"
    )

st.info(
    "2005년 이전 데이터는 곡선을 만드는 데만 사용하고, "
    "2005년 이후 데이터는 학습에 사용하지 않은 채 성능 평가에만 사용합니다."
)


# =========================================================
# X, y 설정
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
# 학습 및 테스트 평가
# =========================================================
results = []

predictions = {}

for name, model in models.items():

    # 반드시 훈련 데이터만 사용하여 학습
    model.fit(
        X_train,
        y_train
    )

    # 학습에 사용하지 않은 테스트 데이터로 예측
    test_prediction = model.predict(
        X_test
    )

    predictions[name] = test_prediction

    # 테스트 데이터에서 평균적으로 몇 도 빗나가는지
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
        "테스트 평균 오차(MAE)": mae,
        "2050년 예상 기온(℃)": prediction_2050
    })


# =========================================================
# 결과 표
# =========================================================
result_df = pd.DataFrame(results)


st.subheader("🎯 곡선별 테스트 성능과 2050년 예측")


display_df = result_df.copy()

display_df["테스트 평균 오차(MAE)"] = (
    display_df["테스트 평균 오차(MAE)"]
    .map(lambda x: f"{x:.2f} ℃")
)

display_df["2050년 예상 기온(℃)"] = (
    display_df["2050년 예상 기온(℃)"]
    .map(lambda x: f"{x:.2f} ℃")
)


st.table(display_df)


st.caption(
    "MAE가 작을수록 테스트 데이터의 실제 기온을 평균적으로 더 가깝게 예측한 것입니다."
)


# =========================================================
# 가장 좋은 모델
# =========================================================
best_model = result_df.loc[
    result_df["테스트 평균 오차(MAE)"].idxmin(),
    "모델"
]

best_mae = result_df.loc[
    result_df["테스트 평균 오차(MAE)"].idxmin(),
    "테스트 평균 오차(MAE)"
]

st.success(
    f"테스트 데이터에서 평균 오차가 가장 작은 모델은 "
    f"**{best_model}**이며, 평균적으로 약 **{best_mae:.2f}℃** 빗나갔습니다."
)


# =========================================================
# 실제 데이터 + 곡선 그래프
# =========================================================
st.subheader("📈 훈련 데이터와 테스트 데이터, 그리고 곡선")


fig = go.Figure()


# 훈련 데이터
fig.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["평균기온"],
        mode="markers",
        name="훈련 데이터",
        marker=dict(
            size=5
        ),
        hovertemplate=
            "연도: %{x}<br>"
            "평균기온: %{y:.2f}℃"
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
            "실제기온: %{y:.2f}℃"
            "<extra></extra>"
    )
)


# 1908~2050까지 부드러운 곡선을 그리기 위한 연도
curve_years = np.linspace(
    yearly["연도"].min(),
    2050,
    500
)

curve_X = pd.DataFrame({
    "경과연수": curve_years - 1908
})


# 각 곡선 표시
for name, model in models.items():

    curve_prediction = model.predict(
        curve_X
    )

    fig.add_trace(
        go.Scatter(
            x=curve_years,
            y=curve_prediction,
            mode="lines",
            name=f"{name} 곡선",
            hovertemplate=
                "연도: %{x:.0f}<br>"
                "예상기온: %{y:.2f}℃"
                "<extra></extra>"
        )
    )


# 2005년 기준선
fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="2005년: 학습 → 테스트"
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
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


col1, col2, col3 = st.columns(3)

for col, (name, model) in zip(
    [col1, col2, col3],
    models.items()
):

    x_2050 = pd.DataFrame({
        "경과연수": [2050 - 1908]
    })

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
with st.expander("📋 테스트 데이터별 실제값과 예측값 보기"):

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
# 설명
# =========================================================
st.subheader("💡 해석")

st.write(
    """
- **1차 곡선**은 직선이므로 전체적인 증가·감소 추세만 표현합니다.
- **3차 곡선**은 직선보다 유연하여 데이터의 굽은 추세를 표현할 수 있습니다.
- **9차 곡선**은 훈련 데이터의 복잡한 변화까지 따라갈 수 있지만, 
  훈련 데이터에 지나치게 맞춰지면 새로운 테스트 데이터에서 오차가 커질 수 있습니다.
- 따라서 어떤 곡선이 좋은지는 훈련 데이터에 얼마나 잘 맞는지가 아니라, 
  **학습에 사용하지 않은 테스트 데이터에서 얼마나 정확한지**로 판단해야 합니다.
"""
)
```


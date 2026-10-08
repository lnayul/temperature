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


# =========================================================
# 데이터 표
# =========================================================
with st.expander("📋 사용된 연도별 데이터 보기"):

    st.dataframe(
        yearly[
            [
                "연도",
                "관측일수",
                "평균기온",
                "예측기온"
            ]
        ].style.format({
            "평균기온": "{:.2f}",
            "예측기온": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

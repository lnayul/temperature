```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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
    "서울의 연평균기온 데이터를 이용하여 "
    "기온의 변화 추세를 분석하고 미래의 기온을 예측합니다."
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
    df = df[
        df["연도"] <= 2025
    ]

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

    st.error(
        "데이터를 불러오는 중 오류가 발생했습니다."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# 기본 데이터 정보
# =========================================================

start_year = int(
    yearly["연도"].min()
)

end_year = int(
    yearly["연도"].max()
)

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

    st
```

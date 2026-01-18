import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# 1. 한글 폰트 설정 (윈도우 기준)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 2. 데이터 불러오기
try:
    df = pd.read_csv("geumcheon_apt_2024_cleaned.csv")
    print("✅ 데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("❌ 파일을 찾을 수 없습니다. main.py를 먼저 실행해 주세요.")
    exit()

# 그래프를 그릴 도화지 준비 (3개의 그래프)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- 그래프 1: 동네별 평균 거래가 ---
sns.barplot(x='거래금액', y='법정동', data=df, ax=axes[0], palette='viridis')
axes[0].set_title('금천구 동네별 평균 아파트 가격')
axes[0].set_xlabel('평균 거래금액(만원)')

# --- 그래프 2: 전용면적과 거래금액의 상관관계 ---
sns.scatterplot(x='전용면적', y='거래금액', data=df, ax=axes[1], alpha=0.5)
axes[1].set_title('전용면적 vs 거래금액')
axes[1].set_xlabel('전용면적(㎡)')
axes[1].set_ylabel('거래금액(만원)')

# --- 그래프 3: 아파트 나이에 따른 가격 분포 ---
sns.regplot(x='아파트나이', y='거래금액', data=df, ax=axes[2], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[2].set_title('아파트 나이 vs 거래금액')
axes[2].set_xlabel('아파트 나이')
axes[2].set_ylabel('거래금액(만원)')

plt.tight_layout()
plt.show()
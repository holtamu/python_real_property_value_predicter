## 3. 텐서플로 모델링
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
try:
    df = pd.read_csv("geumcheon_apt_2024_cleaned.csv")
    print("✅ 데이터를 불러왔습니다.")
except FileNotFoundError:
    print("❌ 데이터 파일이 없습니다. main.py를 먼저 실행하세요.")
    exit()

# 2. [수정] 특징(X)과 정답(y) 분리
# 법정동(글자)을 0과 1로 변환하는 원-핫 인코딩을 수행합니다.
df_encoded = pd.get_dummies(df, columns=['법정동'])

# 입력 데이터: 전용면적, 아파트나이 + 법정동_가산동, 법정동_독산동, 법정동_시흥동 등
# 출력 데이터: 거래금액
# filter를 써서 필요한 컬럼만 X로 가져옵니다.
X = df_encoded.filter(regex='전용면적|아파트나이|법정동_').values
y = df_encoded['거래금액'].values

# 3. 데이터셋 분할 (학습용 80%, 테스트용 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 데이터 스케일링 (표준화)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. [수정] 인공신경망 모델 설계
# input_shape를 고정된 (2,)가 아니라 X의 컬럼 개수에 맞춰 자동으로 설정하게 바꿨습니다.
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)), 
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

# 6. 모델 설정 (컴파일)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 7. 인공지능 학습 시작
print("\n🤖 법정동 정보를 포함하여 인공지능 학습 중...")
history = model.fit(
    X_train_scaled, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2,
    verbose=0 
)
print("✨ 학습 완료!")

# 8. 학습 결과 시각화 (Loss 그래프)
# --- 윈도우 한글 폰트 설정 추가 ---
plt.rc('font', family='Malgun Gothic') # 맑은 고딕 설정
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
# ---------------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='학습 손실 (Train Loss)')
plt.plot(history.history['val_loss'], label='검증 손실 (Val Loss)')
plt.title('모델 학습 과정 (Loss)')
plt.xlabel('반복 횟수 (Epochs)')
plt.ylabel('손실 값 (MSE)')
plt.legend()
plt.show()

# 9. 모델 평가
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n📊 모델 평가 결과 (MAE): 평균 약 {round(mae, 2)}만원 정도의 오차가 발생합니다.")


# 10. 사용자 입력형 실제 예측 테스트

print("\n" + "="*50)
print("🏠 금천구 아파트 가격 예측 시뮬레이터")
print("="*50)

# --- 1. 전용면적 선택 ---
unique_areas = sorted(df['전용면적'].unique())
print("\n[1] 전용면적을 선택하세요 (㎡):")
for i, area in enumerate(unique_areas, 1):
    print(f"{i}. {area}㎡")
area_choice = int(input("번호 입력 >> ")) - 1
input_area = unique_areas[area_choice]

# --- 2. 아파트 나이 선택 ---
unique_ages = sorted(df['아파트나이'].unique())
print("\n[2] 아파트 나이(건축년도 기준)를 선택하세요:")
for i, age in enumerate(unique_ages, 1):
    print(f"{i}. {age}년")
age_choice = int(input("번호 입력 >> ")) - 1
input_age = unique_ages[age_choice]

# --- 3. 법정동 선택 ---
unique_dongs = sorted(df['법정동'].unique())
print("\n[3] 법정동(동네)을 선택하세요:")
for i, dong in enumerate(unique_dongs, 1):
    print(f"{i}. {dong}")
dong_choice = int(input("번호 입력 >> ")) - 1
input_dong = unique_dongs[dong_choice]

# --- 4. 데이터 변환 및 예측 ---
# 원-핫 인코딩된 컬럼 순서에 맞춰서 0과 1의 리스트를 만듭니다.
test_columns = df_encoded.filter(regex='전용면적|아파트나이|법정동_').columns
dong_features = [1 if f"법정동_{input_dong}" == col else 0 for col in test_columns if "법정동_" in col]

# [면적, 나이, 가산동(0), 독산동(0), 시흥동(1) ...] 형태로 조합
sample_data = np.array([[input_area, input_age] + dong_features])
sample_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_scaled, verbose=0)

# --- 5. 최종 결과 출력 ---
print("\n" + "결과 분석 중..." + "."*10)
print(f"\n✅ 선택하신 조건:")
print(f"📍 위치: {input_dong} | 면적: {input_area}㎡ | 나이: {input_age}년")
print("-" * 50)
print(f"💰 인공지능 예측 거래가: 약 {round(prediction[0][0], -1):,} 만원")
print("="*50)

# =========================================
# --- [Figure_2.png] 실제값 vs AI 예측값 비교 그래프 ---
print("\n📊 모델 성능 검증 그래프를 생성합니다...")

# 테스트 데이터에 대한 예측 수행
y_pred = model.predict(X_test_scaled, verbose=0).flatten()

plt.figure(figsize=(8, 8))

# 산점도: 실제값(x)과 예측값(y)
plt.scatter(y_test, y_pred, alpha=0.5, color='royalblue', label='예측 데이터')

# 완벽한 예측을 의미하는 대각선 (y=x)
max_val = max(max(y_test), max(y_pred))
min_val = min(min(y_test), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Perfect Prediction')

plt.title(f'실제 거래가 vs AI 예측가 비교\n(평균 오차: 약 {round(mae, 2)}만원)')
plt.xlabel('실제 거래금액 (만원)')
plt.ylabel('AI 예측 금액 (만원)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('Figure_2.png') # Figure_2로 저장
plt.show()
print("✅ Figure_2.png (성능 검증 그래프) 저장 완료")
# =========================================


# 11. Gemini 연동하기
# gemini_analysis.py에서 함수 불러오기
from gemini_analysis import get_gemini_report
from save_image import save_text_as_image

# 불러온 함수 실행
report = get_gemini_report(input_dong, input_area, input_age, prediction.item())

# 리포트 사진으로 저장
save_text_as_image(report)

# 3. 결과 출력
print("\n" + "="*50)
print("🏠 Gemini 전문가 분석 리포트")
print("-" * 50)
print(report)
print("="*50)
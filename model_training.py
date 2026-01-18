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

# 10. [수정] 실제 예측 테스트
# 예: 전용면적 84㎡, 아파트 나이 10년, 시흥동(세 번째 동네라고 가정) 아파트의 예상 가격은?
# 원-핫 인코딩 순서에 맞춰서 데이터를 넣어줘야 합니다.
# [면적, 나이, 법정동_가산동(0), 법정동_독산동(0), 법정동_시흥동(1)] 형태 예시:
# *주의: 실제 데이터의 동네 순서에 따라 1의 위치가 달라질 수 있습니다.
test_columns = df_encoded.filter(regex='전용면적|아파트나이|법정동_').columns
print(f"\n입력 순서 확인: {list(test_columns)}")

# 시흥동(법정동_시흥동=1)을 가정하고 테스트 데이터를 만듭니다.
# 보통 가산, 독산, 시흥 순이므로 0, 0, 1로 넣어봅니다.
sample_data = np.array([[84, 10, 0, 0, 1]]) 
sample_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_scaled)

print(f"🏠 예측 결과: {list(test_columns)} 조건의 예상가는 약 {round(prediction[0][0], 2)}만원입니다.")
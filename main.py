## 2. 데이터 전처리
import pandas as pd
import time
import os
from dotenv import load_dotenv  # .env 파일을 읽어오는 도구
from data_collector import get_apartment_data

# 1. .env 파일의 환경변수 로드
load_dotenv()

# 2. 로드된 환경변수에서 키 가져오기
# .env에 적은 변수명과 똑같이 'DATA_API_KEY'라고 적어줍니다.
MY_KEY = os.getenv("DATA_API_KEY") 

GEUMCHEON_CODE = "11545"
TARGET_YEAR = "2024"

def main():
    print("--- 프로그램이 정상적으로 시작되었습니다! ---") # 이 문구가 뜨는지 확인하세요.
    # 키가 정상적으로 로드되었는지 확인
    if not MY_KEY:
        print("❌ 에러: .env 파일에서 API 키를 찾을 수 없습니다.")
        print("파일 이름이 '.env'인지, 내부에 'DATA_API_KEY=...'가 있는지 확인하세요.")
        return
    else:
        print(f"🔑 API 키 로드 성공 (앞부분: {MY_KEY[:5]})")

    all_data = []
    
    print(f"🚀 {TARGET_YEAR}년 금천구 아파트 실거래 데이터 수집 시작...")
    print(f"🔑 키 로드 완료: {MY_KEY[:5]}*** (보안 처리됨)")

    # 1월부터 6월까지 수집 반복
    for month in range(1, 7):
        deal_ymd = f"{TARGET_YEAR}{month:02d}"
        print(f"📅 {deal_ymd} 수집 중...", end=" ", flush=True)
        
        # 불러온 MY_KEY를 인자로 전달
        monthly_items = get_apartment_data(MY_KEY, GEUMCHEON_CODE, deal_ymd)
        
        if monthly_items:
            all_data.extend(monthly_items)
            print(f"({len(monthly_items)}건 완료)")
        else:
            print("(데이터 없음)")
        
        time.sleep(0.5)

    if not all_data:
        print("❌ 수집된 데이터가 없습니다.")
        return

    # 데이터프레임 변환 및 전처리
    df = pd.DataFrame(all_data)
    
    print("\n🧹 데이터 전처리 중...")
    df['거래금액'] = df['거래금액'].str.replace(',', '').astype(int)
    df['전용면적'] = pd.to_numeric(df['전용면적'], errors='coerce')
    df['건축년도'] = pd.to_numeric(df['건축년도'], errors='coerce')
    df['아파트나이'] = 2026 - df['건축년도']
    df = df.dropna()

    # 최종 CSV 저장
    output_file = f"geumcheon_apt_{TARGET_YEAR}_cleaned.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print("-" * 30)
    print(f"✨ 수집 및 전처리 완료! 파일명: {output_file}")
    print(f"📊 총 수집 건수: {len(df)}건")

if __name__ == "__main__":
    main()
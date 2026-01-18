## 4. Gemini 연동
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
# 1. API 설정 (발급받은 키를 입력하세요)
GOOGLE_API_KEY = os.getenv("YOUR_GEMINI_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_gemini_report(input_dong, input_area, input_age, predicted_price):
    """
    예측 결과와 조건을 바탕으로 Gemini에게 분석 리포트를 요청합니다.
    """
    
    # Gemini에게 보낼 프롬프트(질문) 구성
    prompt = f"""
    당신은 대한민국 부동산 전문가입니다. 
    최근 딥러닝 인공지능 모델이 분석한 아래 부동산 실거래 예측 데이터를 바탕으로 전문적인 리포트를 작성해 주세요.

    [분석 데이터]
    - 위치: 서울특별시 금천구 {input_dong}
    - 전용면적: {input_area}㎡
    - 아파트 나이: {input_age}년 (건축 후 경과 년수)
    - AI 예측 적정가: 약 {round(predicted_price, -1):,} 만원

    [리포트 포함 내용]
    1. 해당 매물의 가격 적정성 평가
    2. {input_dong} 지역의 최근 부동산 시장 특징 (금천구 특성 반영)
    3. 실거주 및 투자 관점에서의 조언
    4. 향후 해당 조건의 아파트 가격에 영향을 줄만한 요인

    문체는 신뢰감 있고 친절한 전문가 말투(~입니다)로 작성해 주세요.
    """

    print("\n🤖 Gemini가 전문가 리포트를 생성 중입니다...")
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini 연동 중 오류가 발생했습니다: {e}"

# (테스트용 실행 코드)
report = get_gemini_report("시흥동", 84, 10, 55980)
print(report)
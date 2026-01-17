import requests
import pandas as pd
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv


def get_apartment_data(service_key, lawd_cd, deal_ymd):
    """특정 지역과 연월의 아파트 실거래가 리스트를 반환합니다."""
    url = 'https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev'
    request_url = f"{url}?serviceKey={service_key}&LAWD_CD={lawd_cd}&DEAL_YMD={deal_ymd}&numOfRows=1000"
    
    try:
        response = requests.get(request_url, timeout=15)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            res_code = root.find('.//resultCode').text
            
            if res_code in ['00', '000']:
                item_list = []
                for item in root.findall('.//item'):
                    data = {
                        '아파트': item.findtext('aptNm'),
                        '거래금액': item.findtext('dealAmount'),
                        '전용면적': item.findtext('excluUseAr'),
                        '법정동': item.findtext('umdNm'),
                        '층': item.findtext('floor'),
                        '건축년도': item.findtext('buildYear'),
                        '거래년': item.findtext('dealYear'),
                        '거래월': item.findtext('dealMonth'),
                        '거래일': item.findtext('dealDay')
                    }
                    item_list.append(data)
                return item_list
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

# --- 단독 실행 테스트용 ---
if __name__ == "__main__":
    load_dotenv()
    MY_KEY = os.getenv("DATA_API_KEY") # 파일에서 키를 몰래 읽어옴
    # MY_KEY = "YOUR_API-KEY"
    REGION_CODE = "11545" # 금천구
    DATE = "202401"
    df = get_apartment_data(MY_KEY, REGION_CODE, DATE)

    if df:
        print("✅ 모듈 테스트 성공! 데이터 1건 예시:", df[0])
    else:
        print("❌ 모듈 테스트 실패")
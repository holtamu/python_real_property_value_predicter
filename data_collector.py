import requests
import pandas as pd
import xml.etree.ElementTree as ET

def get_apartment_data(service_key, lawd_cd, deal_ymd):
    # 1. 기본 주소
    url = 'https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev'
    
    # 2. 주소 조립 (인증키를 주소에 직접 포함)
    # 포털 미리보기에서 성공했던 그 'Encoding' 키를 그대로 사용하세요.
    request_url = f"{url}?serviceKey={service_key}&LAWD_CD={lawd_cd}&DEAL_YMD={deal_ymd}"
    
    try:
        # 주소 문자열을 그대로 사용하여 호출 (params 옵션을 쓰지 않음)
        response = requests.get(request_url, timeout=15)
        
        # 서버 응답을 바로 출력해서 확인
        print("\n--- 서버 응답 확인 ---")
        if response.status_code == 200:
            print("✅ 서버 연결 성공!")
        else:
            print(f"❌ 서버 연결 실패 (코드: {response.status_code})")
        
        # XML 파싱
        root = ET.fromstring(response.content)

        # 결과 코드 확인 (00 또는 000이 정상)
        res_code = root.find('.//resultCode').text
        res_msg = root.find('.//resultMsg').text
        
        # '00' 또는 '000'이면 성공으로 간주하도록 수정
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
                        '거래일': item.findtext('dealDay'),
                        '지번': item.findtext('jibun')
                    }
                item_list.append(data)
            
            # 만약 데이터가 성공적으로 파싱되었다면 데이터프레임 반환
            return pd.DataFrame(item_list)
        else:
            print(f"❌ API 오류 메시지: {res_msg} ({res_code})")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        return pd.DataFrame()

# --- 실행 ---
# 포털 '미리보기'에서 성공했을 때 사용했던 그 키를 복사해서 넣으세요!
MY_KEY = "공공데이터포털 API 키" 
REGION_CODE = "11545" # 금천구
DATE = "202401"

df = get_apartment_data(MY_KEY, REGION_CODE, DATE)

if not df.empty:
    print(f"✅ 성공! {len(df)}건의 데이터를 가져왔습니다.")
    print(df[['아파트', '거래금액', '법정동']].head())
else:
    print("❌ 여전히 에러가 난다면 키를 'Decoding' 키로 바꿔서 한 번만 더 해보세요.")
import requests

# 사용자님의 키를 직접 넣었습니다.
service_key = "1f907abf46073183fa9bb09421650f71e94feff1ea6d61f447379ddbb4d8f9ed"
url = f"http://apis.data.go.kr/1613000/RTMSOBJSvc/getRTMSDataSvcAptTrade?serviceKey={service_key}&LAWD_CD=11545&DEAL_YMD=202312"

try:
    response = requests.get(url, timeout=15)
    print(f"상태 코드: {response.status_code}")
    print("응답 내용:")
    print(response.text)
except Exception as e:
    print(f"오류 발생: {e}")

print(url)
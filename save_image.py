from PIL import Image, ImageDraw, ImageFont

# --- 리포트를 PNG 이미지로 저장하는 로직 ---
def save_text_as_image(text, filename="Figure.png"):
    # 폰트 설정 (윈도우 맑은 고딕 경로)
    try:
        font = ImageFont.truetype("malgun.ttf", 20)
        title_font = ImageFont.truetype("malgunbd.ttf", 28)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # 이미지 여백 및 줄바꿈 설정
    margin = 40
    line_spacing = 10
    width = 1800
    
    # 텍스트 줄바꿈 처리
    lines = []
    for line in text.split('\n'):
        # 너무 긴 문장은 잘라서 여러 줄로 만듦 (한글 기준 약 40자)
        if len(line) > 100:
            for i in range(0, len(line), 100):
                lines.append(line[i:i+100])
        else:
            lines.append(line)

    # 이미지 높이 계산 (줄 수에 따라 유동적으로 변경)
    line_height = font.getbbox("가")[3] + line_spacing
    height = margin * 2 + len(lines) * line_height + 100

    # 배경 이미지 생성 (흰색)
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 제목 및 본문 쓰기
    draw.text((margin, margin), "🏠 Gemini 부동산 분석 리포트", fill=(0, 0, 0), font=title_font)
    
    y_text = margin + 80
    for line in lines:
        draw.text((margin, y_text), line, fill=(50, 50, 50), font=font)
        y_text += line_height

    # 이미지 저장
    img.save(filename)
    print(f"✅ {filename} 저장이 완료되었습니다!")
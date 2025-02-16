### === 0. Import Libraries ===
import streamlit as st 

import os
import torch
import textwrap
import hashlib
import random
import io
import tempfile
import uuid

from deep_translator import GoogleTranslator 
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from openai import OpenAI

from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.utils import ImageReader


### === 1. Configs ===
Image.MAX_IMAGE_PIXELS = None

FONT_PATH = os.getenv("FONT_PATH", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf") 
pdfmetrics.registerFont(TTFont("NanumGothic", FONT_PATH))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_DEFAULT_API_KEY")
client = OpenAI(api_key = OPENAI_API_KEY)

## Hugging Face 모델 로드 
@st.cache_resource
def load_models():
    """
    Hugging Face 모델 & 토크나이저 로드
    - 캐싱을 활용하여 로딩 속도 증가
    """
    # Image Captioning
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    # Story Generator
    MODEL_DIR = os.getenv("MODEL_DIR", "YOUR_MODEL_DIR")  # Fine-tuned koT5 

    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if not os.path.exists(MODEL_DIR) or missing_files:
        raise ValueError(f"🚨 모델 경로가 올바르지 않거나 파일이 누락되었습니다: {MODEL_DIR}\n❌ 누락된 파일: {missing_files}")

    t5_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
    
    return blip_processor, blip_model, t5_tokenizer, t5_model

blip_processor, blip_model, t5_tokenizer, t5_model = load_models()


### === 2. utils ===
@st.cache_data
def save_temp_file(uploaded_file):
    """업로드된 파일을 임시 저장"""
    temp_dir = "/content/temp_images"
    os.makedirs(temp_dir, exist_ok=True) 
    
    file_ext = os.path.splitext(uploaded_file.name)[-1] 
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}{file_ext}")

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return temp_path

@st.cache_data
def translate_text(text, src="en", dest="ko"):
    """문자열을 번역하는 함수 (deep-translator 사용)"""
    try:
        return GoogleTranslator(source=src, target=dest).translate(text)
    except Exception as e:
        return text  

## 📷 이미지 캡셔닝 (BLIP 모델 활용)
@st.cache_data
def generate_korean_caption(image_path):
    """
    이미지에서 캡션 생성 후 한국어로 번역 (googletrans 사용)
    """
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = blip_processor(image, return_tensors="pt").to(device)
        outputs = blip_model.generate(**inputs)
        english_caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
        korean_caption = translate_text(english_caption, src="en", dest="ko")
        image.close()
        return korean_caption
    except Exception as e:
        st.warning(f"🚨 번역 실패: {e}, 원본 영어 캡션 반환.")
        return english_caption


## 🎭 보조 정보 자동 추천
@st.cache_data
def suggest_story_elements(caption):
    """
    GPT-4 API를 활용하여 보조적 정보 추천
    """
    prompt = (
        f"다음 이미지 설명을 기반으로 이야기에 적합한 배경, 행동, 감정 요소를 추천해줘.\n"
        f"각 요소는 3~5개씩 추천하고, 불필요한 설명 없이 쉼표(,)로 구분된 키워드 형태로만 출력해.\n"
        f"\n설명: {caption}\n"
        f"\n출력 예시 (배경: 키워드1, 키워드2, 키워드3 / 행동: 키워드1, 키워드2 / 감정: 키워드1, 키워드2, 키워드3):"
    )
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "당신은 창의적인 이야기 도우미입니다."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-4o",
        max_tokens=50
    )
    return response.choices[0].message.content


## input text 생성 
@st.cache_data
def generate_input_text(caption, name, i_action, classification, additional_info):
    seed_source = f"{caption}{name}{i_action}"
    seed_value = int(hashlib.md5(seed_source.encode()).hexdigest(), 16) % (10 ** 8)
    random.seed(seed_value) 

    required_fields = [
        f"<caption> {caption}",
        f"<name> {name.strip()}",
        f"<i_action> {i_action.strip()}",
        f"<classification> {classification.strip()}"
    ]

    optional_tokens = [
        f"{token} {additional_info.get(field, '<empty>') if additional_info else '<empty>'}"
        for field, token in {
            "character": "<character>",
            "setting": "<setting>",
            "action": "<action>",
            "feeling": "<feeling>",
            "causalRelationship": "<causalRelationship>",
            "outcomeResolution": "<outcomeResolution>",
            "prediction": "<prediction>"
        }.items()
    ]

    all_tokens = required_fields + optional_tokens
    random.shuffle(all_tokens)

    return "generate: " + " ".join(all_tokens)


## 📝 스토리 생성
special_tokens = [
    "<caption>", "<name>", "<i_action>", "<classification>", "<character>",
    "<setting>", "<action>", "<feeling>", "<causalRelationship>",
    "<outcomeResolution>", "<prediction>", "<empty>"
]
special_token_ids = t5_tokenizer.convert_tokens_to_ids(special_tokens)

def special_token_masking(input_ids, attention_mask, special_token_ids):
    special_token_ids = torch.tensor(special_token_ids, device=input_ids.device)
    mask_indices = (input_ids.unsqueeze(-1) == torch.tensor(special_token_ids).clone().detach()).any(dim=-1)
    attention_mask[mask_indices] = 0
    return attention_mask

@st.cache_data
def generate_story(input_text):
    inputs = t5_tokenizer(input_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt").to(device)
    inputs["attention_mask"] = special_token_masking(inputs["input_ids"], inputs["attention_mask"], special_token_ids).to(device)
    
    with torch.no_grad():
        output_ids = t5_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], 
            max_length=128,
            num_beams=3,
            length_penalty=0.8,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


@st.cache_data
def save_story_as_pdf(story_data):
    """이야기를 PDF로 저장 (한글 깨짐 방지)"""
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer)
    
    c.setFont("NanumGothic", 12)

    for row in story_data:
        wrapped_text = textwrap.wrap(row["story"], width=40) 
        y_position = 750
        image_path = row.get("imgpath")

        if not image_path or not os.path.exists(image_path):
            image_path = "/content/default_image.jpg"
            if not os.path.exists(image_path):
                raise ValueError(f"🚨 오류 발생: 이미지 파일이 존재하지 않습니다! → {image_path}")

        try:
            img = ImageReader(image_path)
            c.drawImage(img, 50, 500, width=400, height=300, preserveAspectRatio=True)
            y_position = 480
        except Exception as e:
            raise RuntimeError(f"🚨 이미지 로드 오류 발생: {e}")

        for line in wrapped_text:
            if y_position < 50:
                c.showPage()
                c.setFont("NanumGothic", 12) 
                y_position = 750
            c.drawString(50, y_position, line)
            y_position -= 20

        c.showPage()
        c.setFont("NanumGothic", 12)

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

    
    
### === 3. Streamlit UI 구성 ===
st.title("📖 AI 동화책 생성기")
st.write("이미지를 업로드하면 AI가 자동으로 이야기를 만들어 줍니다 ✨")

'''
- 사용자 이미지 업로드
- captioning 수행
- 보조 정보 추천
'''
uploaded_files = st.file_uploader("📸 이미지를 업로드하세요(최대 10장)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    user_inputs = []

    if "uploaded_image_paths" not in st.session_state:
        st.session_state["uploaded_image_paths"] = {}

    for idx, uploaded_file in enumerate(uploaded_files):
        img_path = save_temp_file(uploaded_file) 

        st.session_state["uploaded_image_paths"][uploaded_file.name] = img_path

        image = Image.open(img_path).convert("RGB")
        st.image(image, caption=f"업로드된 이미지{idx + 1}", use_container_width=True)
        
        with st.spinner("📸 캡션 생성 중..."):
            caption = generate_korean_caption(img_path)       
        st.write(f"📌 **자동 생성된 캡션:** {caption}")
        
        st.subheader(f"💬 보조 정보 입력{idx + 1}")
        
        name = st.text_input("📍 객체 이름", placeholder="ex) 강아지", key=f"name_{idx + 1}")
        i_action = st.text_input("🎭 객체의 행동", placeholder="ex) 뛰다", key=f"i_action_{idx + 1}")
        classification = st.selectbox("📚 이야기 분류", ["의사소통", "자연탐구", "사회관계", "예술경험", "신체운동_건강"], key=f"classification_{idx + 1}")
        
        suggested_info = suggest_story_elements(caption)
        st.write(f"🔹 추천 정보({idx + 1}): {suggested_info}")
        
        additional_info = {
            "character": st.text_input("👤 캐릭터 정보 (선택)", placeholder="ex) 주인공, 친구", key=f"character_{idx + 1}"),
            "setting": st.text_input("🏞️ 배경 정보 (선택)", placeholder="ex) 숲속, 병원", key=f"setting_{idx + 1}"),
            "action": st.text_input("⚡ 행동 설명 (선택)", placeholder="ex) 놀란다, 고민하다", key=f"action_{idx + 1}"),
            "feeling": st.text_input("😃 감정 상태 (선택)", placeholder="ex) 기뻐하다, 놀라다", key=f"feeling_{idx + 1}"),
            "causalRelationship": st.text_input("🔗 인과 관계 (선택)", placeholder="ex) 친구를 만났다 → 함께 모험을 떠났다", key=f"causalRelationship_{idx + 1}"),
            "outcomeResolution": st.text_input("🎯 결과 (선택)", placeholder="ex) 온 가족이 기뻐했다", key=f"outcomeResolution_{idx + 1}"),
            "prediction": st.text_input("🔮 예측 (선택)", placeholder="ex) 이후 왕국을 구한다", key=f"prediction_{idx + 1}")
        }

        user_inputs.append({
            "imgpath": img_path,
            "caption": caption,
            "name": name,
            "i_action": i_action,
            "classification": classification,
            "additional_info": additional_info
        })

    if st.button("🚀 이야기 생성"):
        stories = []
        
        for user_input in user_inputs:
            input_text = generate_input_text(
                user_input["caption"], 
                user_input["name"], 
                user_input["i_action"], 
                user_input["classification"], 
                user_input["additional_info"]
            )
            story = generate_story(input_text) 
            stories.append({"imgpath": user_input["imgpath"], "story": story})

        st.success("이야기가 생성되었습니다! ✅")

        pdf_buffer = save_story_as_pdf(stories)
        st.download_button("📥 PDF로 저장", data=pdf_buffer, file_name="storybook.pdf", mime="application/pdf")
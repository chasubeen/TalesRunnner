# **🔮 TalesRunner**

## **1. 프로젝트 소개**
- 사용자에게 몇 장의 이미지를 입력받으면, 이에 해당하는 스토리라인을 구성 후 그에 맞는 이야기를 `pdf`로 출력해 주는, 일종의 ‘그림책’을 만들어주는 프로젝트입니다.  
  <img src = "https://github.com/user-attachments/assets/1b135c0c-f546-4b8d-9fa1-9f6f64c6bc65" width = 600 height = 250>
    
    

### **🛠 전체 framework 소개**
<img src = "https://github.com/user-attachments/assets/7c303b82-4495-4cfd-8dcc-f7b8d45173d6" width = 500 height = 700>  
  

**[설명]**
- 빨/파/초: 이미지-텍스트 pair를 구분하기 위함  
  (별다른 의미는 없음)  
- 노란색: pre-trained model  
  (학습 x, inference 용도로 활용)  
- 보라색: fine-tuned model  
  (데이터셋을 통해 원하는 목적에 부합하도록 fine-tuning 수행)

**[사용자 입력]**
1. 몇 장의 이미지를 순서대로 입력합니다.(최대 10장)
2. 이후, 각 장면에 해당하는 보조적 정보들을 입력합니다.  
  - 캡션으로부터 키워드 자동 추천 기능 추가
  - 필수 정보) 주인공 이름, 행동, 이야기 분류(5지선다)
  - 선택 정보) 캐릭터, 배경, 행동, 감정, 인과 관계, 결과, 예측

**[모델]**  
1. 사용자가 입력한 이미지를 pre-trained `BLIP` model에 통과시켜 이미지에 해당하는 캡션을 생성합니다.
2. 이후, 캡션과 사용자가 입력한 보조적 정보들을 결합하여 `input_text`를 생성합니다.  
   `<caption> 숲 속의 나비 <setting> 숲 ...`
3. Story Generation 모델은 `input_text`를 기반으로 각 장면에 해당하는 이야기 단락을 생성합니다.
4. 생성된 이야기 단락과 원본 이미지를 다시 연결합니다.
5. 최종 결과물(이야기책)을 파일로 다운받을 수 있도록 `pdf`로 변환합니다.

### **🎯 목표**
- Vision-Language Model(Multi-Modal) 기술을 활용한 프로젝트입니다.  
    - 이미지 캡셔닝 모델을 활용하여 사용자가 입력한 이미지로부터 적절한 캡션을 생성하고, 이를 기반으로 텍스트를 생성/확장  
    - 이야기 구성에 필요한 보조적 정보 또한 결합하여 이야기를 보강  
    ⇒ 이미지와 텍스트의 자연스러운 연계
- pre-trained model 및 API 활용을 통해 효율적인 framework 개발을 도모했습니다.
  - pre-trained BLIP 활용: [[HuggingFace Hub] Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
  - koT5 fine-tuning: [[HuggingFace Hub] wisenut-nlp-team/KoT5-base](https://huggingface.co/wisenut-nlp-team/KoT5-base)
  - 키워드 추출) GPT-4o: [[OpenAI API Key] GPT-4o](https://platform.openai.com/docs/models#gpt-4o)


## **2. 참여자**

| **팀원 1** | **팀원 2** | **팀원 3** | **팀원 4** |
| --- | --- | --- | --- |
| <img src="https://github.com/user-attachments/assets/e44a03e4-de9b-4f73-91f8-9ecac31d76cb" width = 100 height = 100> | <img src = "" width = 100 height = 100> | <img src = "https://github.com/user-attachments/assets/dfca1abb-336c-4463-be89-1271a23438e4" width = 100 height = 100> | <img src = "https://github.com/user-attachments/assets/1b874ffc-13c3-4e81-b408-ad61bdee29db" width = 100 height = 100> |
| [김도은](https://github.com/doeunyy) | [신유진](https://github.com/hineugene) | [우정아](https://github.com/ktde24) | [차수빈](https://github.com/chasubeen) |


## **3. 진행 과정**
### **기간**
- 2025.01.11(토) ~ 2025.02.15(토)

### **세부 일정**  
<img src="https://github.com/user-attachments/assets/57060540-21b8-4ae4-85e4-373f95225ce4" width = 600 height = 400>  

### **역할**

| **이름** | **역할** |
| --- | --- |
| 김도은 | - 주제 구체화<br>- 데이터셋 구축(img → caption)<br>- koT5 fine-tuning(Train & Validation) |
| 신유진 | - 주제 구체화<br>- 데이터셋 구축(annotations 가공)<br> - Inference_streamlit demo(pipeline) |
| 우정아 | - 주제 구체화<br> - 선행 연구 조사(koGPT, koT5)<br> - Story Generator_baseline(`koGPT`)<br> - koT5 Testing(decoding parameter 탐색) |
| 차수빈(**PM**) |- 아이데이션<br> - 선행 연구 조사(koGPT, koT5)<br> - Story Generator_baseline(`koT5`)<br> - Inference_baseline<br> - Inference_streamlit demo(UI) |



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
| <img src="https://github.com/user-attachments/assets/e44a03e4-de9b-4f73-91f8-9ecac31d76cb" width = 100 height = 100> | <img src = "https://github.com/user-attachments/assets/ed8ea73d-d2bf-4c43-b6bc-b67b4a6338be" width = 100 height = 100> | <img src = "https://github.com/user-attachments/assets/dfca1abb-336c-4463-be89-1271a23438e4" width = 100 height = 100> | <img src = "https://github.com/user-attachments/assets/1b874ffc-13c3-4e81-b408-ad61bdee29db" width = 100 height = 100> |
| **[김도은](https://github.com/doeunyy)** | **[신유진](https://github.com/hineugene)** | **[우정아](https://github.com/ktde24)** | **[차수빈](https://github.com/chasubeen)** |


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

## **4. 파일 구조**

```
TalesRunner/
│── data/                           # 데이터셋 구축 code
│   ├── Image_Captioning_BLIP.ipynb # img to caption
│   ├── [Euron7th]_annotation.ipynb # 이야기 보조 정보 추출
│   ├── Merged_Dataset.ipynb        # caption + annotations
│── koT5                            # story generator
│   ├── finetuning.ipynb            # 학습/검증
│   ├── testing.ipynb               # 평가 및 decoding 파라미터 탐색
│── streamlit                       # inference(streamlit demo)
│   ├── app.py                      # inference pipeline 및 UI
│   ├── streamlit_colab.ipynb       # streamlit 실행 파일
```

- 구축한 데이터셋은 공개하지 않습니다.
- Model(fine-tuned koT5): [[Google Drive] model files](https://drive.google.com/drive/folders/1A-ZIijs40xKSFf-AIDtNEDz4xhrDskzg?usp=sharing)  
※ koGPT는 fine-tuning 중간 과정에서 튜닝을 중단하였기에 코드는 공개하지 않습니다.  
    - `.json` 파일들: tokenizer 및 model 설정 관련 파일  
    - `model.safetensors`: model 파일  
      > HuggingFace `transformers` 라이브러리 활용 시 json 파일들 및 model file을 동일 경로(`MODEL_DIR`)에 위치시킨 후 다음과 같이 load하면 됩니다.
    
      ```python
      from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
      
      t5_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
      t5_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
      ```
      
    - `kot5_checkpoint_final.pt`: model checkpoint 파일

## **5. 프로젝트 흐름 및 사용 기술**
### **1) 데이터셋 준비/구축**
- AiHub 공개 데이터셋 활용: [Ai Hub_동화 삽화 생성 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71695)
    - 총 50001건의 데이터(train: validation = 8:1)
    - 이미지(`.jpg`) 및 annotation 파일(`.json`)이 쌍으로 존재
- 이미지 파일: pre-trained BLIP에 통과시켜 캡션 생성
- annotations 파일: `json` 파일에서 이야기 생성에 도움이 될만한 요소 추출
    - srcText: 이미지에 해당하는 GT 이야기 단락
    - 보조 정보 목록
        |필드|설명|
        | --- | --- |
        | srcText | * 이미지에 해당하는 이야기 단락(GT)<br>* 이후 LM의 input text 가공에 활용<br>* 필수 |
        | name | * 객체의 일반명사<br>* 필수 |
        | i_action | * 객체의 행동, 상태<br>* 필수 |
        | classification | * 의사소통/자연탐구/사회관계/예술경험/신체운동_건강<br>* 필수 |
        | character | * 캐릭터<br>* 선택 |
        | setting | * 셋팅<br>* 선택 |
        | action | * 행동<br>* 선택 |
        | feeling | * 감정<br>* 선택 |
        | causalRelationship | * 인과<br>* 선택 |
        | outcomeResolution | * 결과<br>* 선택 |
        | prediction | * 예측<br>* 선택 |

- 이후 id를 통해 caption과 annotation을 결합하여 하나의 dataset으로 구축
    - `dataset_train.csv`, `dataset_val.csv`

### **2) Story Generator**
- baseline model: [KoGPT-2](https://github.com/SKT-AI/KoGPT2), [KoT5](https://huggingface.co/wisenut-nlp-team/KoT5-base)
    - 두 model 모두 Transformer 기반 모델
    - attention mechanism을 통해 문맥을 이해하고, 문장 내 단어 간 관계를 학습하여 자연스러운 문장 생성 가능
- koGPT-2 vs koT5
    
    | 구분 | KoGPT2 | KoT5 |
    | --- | --- | --- |
    | 모델 구조 | Decoder-only<br>(Causal Language Model) | Encoder-Decoder<br>(Seq2Seq) |
    | 입력과 출력 관계 | 입력을 포함한 상태에서 이어서 생성 | 다를 수 있음 |
    | 입력 순서 | Caption → Context 순서로 고정 | 무작위 순서 허용 |
    | 텍스트 생성 방식 | 한 글자씩 다음 단어 예측 | 전체 의미를 보고 문장 생성 |
    | 새로운 이야기 생성 | 어려움<br>(입력 복사 경향) | 가능 |
    
    ⇒ GPT는 AutoRegressive 방식으로 문장을 생성하기에 입력을 그대로 따라가려는 경향이 강함  
    ⇒ koT5를 최종 baseline model로 선택  
    
- `text` 가공
    - special token을 활용하여 각 정보를 구분
        - **[필수 필드]**
            - 결측치 확인(결측치 허용 x)
            ```python   
            required_fields = ["caption", "name", "i_action", "classification"]
            for field in required_fields:
                assert pd.notna(row[field]) and row[field].strip(), f"Error: '{field}' 필드는 비워둘 수 없습니다."
            ```   
        - **[선택 필드]**
            - 다중 아이템의 경우 콤마(,)로 구분 후 각 항목에 special token부여
            - 결측치가 있는 경우 `<empty>` 토큰 부여
    - caption과 보조적 정보를 포함하여 구성
        - 행별 고유 시드 설정: 각 행의 id(또는 index)를 seed로 설정  
            → 동일한 데이터를 사용하면 언제나 같은 결과가 나오도록 보장  
        - 랜덤 순서 보장: 각 행마다 token의 순서를 랜덤하게 설정  
            → 모델이 token 간 순서나 위치에 지나치게 의존하지 않고, 각 요소의 의미와 역할을 학습하도록 유도  
    - prefix 추가 → 수행해야 하는 task에 대한 지침 제공
    - 최종 input, output sample  
      <img src = "https://github.com/user-attachments/assets/158add5d-550a-4b3f-9e86-09a0d2aa3757" width = 650 height = 250>

- tokenizer/model 확장
    - 학습 시 이야기의 각 요소들을 구분하기 위한 special token을 추가
        ```python
        special_tokens = [
            "<caption>", "<name>", "<i_action>", "<classification>", 
            "<character>", "<setting>", "<action>", "<feeling>", 
            "<causalRelationship>", "<outcomeResolution>", "<prediction>", "<empty>"
        ]
        ```
    - 추가된 token을 tokenizer와 model에 모두 반영
    - special token들은 일종의 placeholder 역할
        - 학습 시 반영 x
        - attention score 계산 시 제외하기 위해 masking 적용
- 학습
    - Bayesian Optimization(`scikit-optimize`)을 활용하여 최적 파라미터 조합 탐색
    - 이후 학습 진행
        - Optimizer: AdamW
            - warmup 및 learning rate scheduler(linear) 활용
        - early stopping 적용
- 이야기 생성
    - decoding parameter(`model.generate()`) 최적화
    - 평가 지표
        - BERTScore: 생성된 문장과 정답 간 의미적 유사도 평가
        - METEOR: 형태소 및 어순 유사도 평가
        - CIDEr: 정답 문장에서 중요한 단어를 포함하는가
        - SPICE: 문장의 논리적 구조 평가   
    ```
    === 최적의 하이퍼파라미터 조합 ===
    num_beams: 3                                                          
    length_penalty: 0.8                                                       
    repetition_penalty: 1.5                                                    
    no_repeat_ngram_size: 3 
    ```

### **3) Streamlit**
- 전체 framework 구축 후 streamlit을 활용하여 demo page 제작
- UI  
  <img src = "https://github.com/user-attachments/assets/ace9549d-6115-4f02-b051-b7d0a29ff126" width = 700 height = 300>
- streamlit demo page 실행
    - [app.py](https://github.com/chasubeen/TalesRunnner/blob/main/streamlit/app.py)및 model file 필요
    - [실행 code(localtunnel 활용)](https://github.com/chasubeen/TalesRunnner/blob/main/streamlit/streamlit_colab.ipynb)
        - pre-trained 딥러닝 모델들을 불러와 추론에 활용하기에, GPU 환경에서 구동하시는 걸 추천드립니다.
        - 로컬 GPU 환경을 활용하실 경우 localhost로 바로 접속 가능합니다.
            - localtunnel 설치 필요 x
            - 마지막 명령어 수정
                ```bash
                streamlit run app.py
                ```
    - 접속
        - 마지막 코드를 실행하시면, localtunnel은 8501 port를 통해 외부에서 접속할 수 있는 임시 주소를 생성합니다.  
          <img src = "https://github.com/user-attachments/assets/c1f54b5f-f329-40d4-96a6-5489ea3cb63b" width = 500 height = 200>    
        - 이후 Tunnel password에 Externel URL 내의 port 번호를 입력합니다.  
            - 예시) 34.16.230.153  
            <img src = "https://github.com/user-attachments/assets/39883bab-98ff-475b-aebb-c9e5d01429cb" width = 500 height = 350>
            
        - password를 입력히시면 위의 UI와 동일한 streamlit page가 나오게 됩니다.  
            (추론 과정은 로컬 GPU 및 Colab에서 구동됩니다.)      
        
      ※ localhost를 활용하시는 경우 local URL로 바로 접속하시면 됩니다.

## **6. 프로젝트 관련 자료**
- [프로젝트_Notion 페이지](https://euron-7th.notion.site/TalesRunner_-9aff24c76c374e9994200342f272e5f6?pvs=4)
- [발표자료](https://github.com/Ewha-Euron/EURON-PROJECT-/blob/master/7%EA%B8%B0/%5BResearch%5D%20%ED%85%8C%EC%9D%BC%EC%A6%88%EB%9F%AC%EB%84%88_%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EA%B8%B0%EB%B0%98%20%EB%8F%99%ED%99%94%20%EC%83%9D%EC%84%B1.pdf)

---
## **회고**
### **김도은**

### **신유진**

### **우정아**

### **차수빈**

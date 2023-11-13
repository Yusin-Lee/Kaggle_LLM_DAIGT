# LLM-DAIGT
2023 Kaggle Detect AI Generated Text

과제 목표 : 사람이 쓴 글과 AI가 생성한 글을 구별

# 1. 기반 지식, 기초 전처리 from Overview & EDA
   - 기존에 주어진 학습 데이터는 단 3개의 AI generated text가 존재 -> 더 많은 데이터가 필요
   - Kaggle 내 여러 Dataset을 결합하여 사용
      - https://www.kaggle.com/datasets/radek1/llm-generated-essays : gpt-3.5-turbo(500개)와 gpt-4(200개)를 사용하여 생성한 700개의 에세이
      - https://www.kaggle.com/datasets/alejopaullier/daigt-external-dataset : FeedBack Prize 3 competition에서 사용된 2421개의 에세이, 그리고 이에 페어링된 2421개의 AI가 생성한 에세이
      - https://www.kaggle.com/datasets/thedrcat/daigt-proper-train-dataset : 캐글 내 다양한 데이터 셋을 결합하여 44155개의 에세이를 구성. 이 중 29,792개는 사람이 쓴 에세이 14,414개는 AI가 생성한 에세이
      - 3개의 데이터 셋을 결합한 후, 데이터 셋 내 중복을 제거
   -  사람이 쓴 에세이와 AI가 생성한 에세이는 글의 길이 측면에서 평균 및 분포가 크게 차이 : AI가 생성한 에세이는 모두 4000자 이하

# 2. PLM 기반의 모델링
   -  베이스 모델은 deberta-v3 모델을 사용
   -  xsmall, base, large에 따라 text length, batch size 등을 조정
   -  efficiency tuning을 위해 LoRA 사용 : r = 8, alpha = 16, module = ['query','value']
   -  cuda의 autocast를 사용하여 MP 적용
   -  accumulation은 Batch size에 따라 full batch가 32 혹은 64가 되도록 조정

# 3. 후처리 방법
   - 4000자 이상은 모두 사람이 쓴 에세이로 간주 -> 효과 ?
   - submission이 0 혹은 1이 아닌 0~1 사이의 실수로 받기 때문에, sigmoid로 처리

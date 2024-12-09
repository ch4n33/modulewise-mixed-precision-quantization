# Requirements
1. huggingface transformers=4.30.1
1. torch=2.4.1+cu121 (depends on hardware version)
2. pandas=2.2.3
3. python=3.10.13
# apply-qat 실행 방법
1. python3 apply-qat/main.py


## apply-qat로 특정 transformer 모델 qat 적용 과정
1. mixed_qat.py를 참조
2. 모델 내 각 모듈에 apply_qat 모듈 적용
  1. word_embeddings
  2. dense
  3. attention.self
3. 생성된 모델을 train.py 등을 이용하여 학습

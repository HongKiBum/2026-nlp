# Project 1. Seq2Seq Num2Text

2026-2 자연어처리 Project 1 제출물입니다. 숫자·영어·한국어 사이의 숫자 읽기를 6개 방향으로 번역하는 Seq2Seq 모델입니다.

| 방향 | 예시 |
|---|---|
| num → en / en → num | `12708` ↔ `twelve thousand seven hundred eight` |
| num → ko / ko → num | `12708` ↔ `만 이천칠백팔` |
| en → ko / ko → en | 영어 ↔ 한국어 |

숫자도 정수(int)로 처리하지 않고 **문자열**로 읽어 한 글자씩 토큰화합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `project01_seq2seq.ipynb` | 데이터 처리 → 학습 → 저장 → 재로드 → 평가 → 테스트 전체 코드와 실행 결과 |
| `num_en.pt` | num ↔ en 모델 |
| `num_ko.pt` | num ↔ ko 모델 |
| `en_ko.pt` | en ↔ ko 모델 |

## 모델

- **구조:** GRU Encoder–Decoder (단방향 1층, Embedding 128 / Hidden 256) + Scaled Dot-Product Attention
- **디코딩:** Greedy
- **모델 수:** 언어쌍별 독립 모델 3개. 한 모델이 양방향을 담당하고, 입력 앞의 목표 언어 태그(`<TO_num>`, `<TO_en>`, `<TO_ko>`)로 방향을 구분합니다.
- **토큰화:** 숫자·한국어는 문자 단위, 영어는 단어 단위
- **학습:** Teacher Forcing, Cross-Entropy(PAD 제외), Adam(lr 1e-3), batch 256, gradient clipping 1.0, 최대 15 epoch, Validation loss 기준 early stopping(patience 4)
- **데이터 분할:** 100,000행 → Train 80,001 / Validation 9,999 / Test 10,000 (seed 42). 단독 `0 / zero / 영` 행이 Validation에만 있어 어휘에서 빠지는 문제가 있어, 학습 전에 이 한 행을 Train으로 옮겼습니다.

## 결과 (Test 10,000행, Exact Match)

| 방향 | 오답 수 | Exact Match |
|---|---:|---:|
| num → en | 5 | 99.95% |
| en → num | 6 | 99.94% |
| num → ko | 3 | 99.97% |
| ko → num | 0 | 100.00% |
| en → ko | 3 | 99.97% |
| ko → en | 2 | 99.98% |

앞뒤 공백을 제거한 뒤 비교한 기준이며, 공백까지 그대로 비교하면 전체 오답은 21건입니다. Epoch별 loss와 Validation 결과, 오답 목록은 노트북 출력에서 확인할 수 있습니다.

## 실행 방법

1. 노트북과 `.pt` 3개, 데이터 파일 `num2txt.txt`, `num2kor.txt`를 같은 폴더에 둡니다. (Colab에서는 실행 중 업로드 창이 뜹니다.)
2. 위에서부터 순서대로 실행합니다.
   - `.pt` 3개가 있으면 학습을 건너뛰고 저장된 모델을 불러옵니다.
   - `.pt`가 하나도 없으면 3개 모델을 처음부터 학습해 저장합니다.
3. 마지막 셀에서 저장된 `.pt`를 새로 불러와 6개 방향을 테스트합니다.

임의 입력 번역 (앞 셀 실행 후):

```python
translate_final('12708', 'num', 'en')
translate_final('만 이천칠백팔', 'ko', 'num')
translate_final('twelve thousand seven hundred eight', 'en', 'ko')
```

언어 코드는 `'num'`, `'en'`, `'ko'`이며, 숫자도 문자열로 입력합니다.

## 필요 패키지

`torch`, `numpy`, `pandas`, `scikit-learn`, `ipython`

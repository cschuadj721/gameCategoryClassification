# gameCategoryClassification

## 디렉토리구조
### gameCategoryClassification (루트 폴더)
###   datasets(하위 폴더)

### **파이선 파일들은루트 폴더에 복사
### **csv파일들은 datasets 폴더에 복사

## .py 파일설명(알파벳 순서)
### addWordCount.py(csv파일에 단어수 추가)
### ConvRNN.py - 단순 Conv RNN 모델
### extractWordRange.py - 단어수 범위별로 csv파일 나누기 (예: 100~200 단어, 500단어이상 등등)
### fileMerge.py - CSV파일 합치기
### finalTraining.py - 최종 학습 모델 (convRNN, 병렬구조, 양방향, recurrent Dropout, Attention)
### predictFromCSVDATA.py - .CSV 파일로 학습완료모델(.h5) 테스트
### predictFromTestData - .npy 테스트 파일로 학습완료모델(.h5) 테스트
### predictFromTextString - 문자열 파일로 학습완료모델(.h5) 테스트
### preprocessing.py - 전처리
### recategorize.py - 학습에 적합한 분류라벨 적용
### reviewFile.py - 자료구조 파악
### RNNConvAttention.py - Conv RNN, Attention 모델 (케라스 버그로 결과물(.h5) 사용 불가)
### RNNConvAttentionEnsemble.py - Conv RNN, Attention, Ensemble 모델 (학습시간 문제로 테스트 불가)


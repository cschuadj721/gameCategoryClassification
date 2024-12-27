# General-purpose libraries
import pandas as pd
import numpy as np

# Scikit-learn for machine learning utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Saving and loading models
import pickle

# TensorFlow for deep learning
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Text preprocessing for English
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# NLTK 자연어 처리 기본 도구 다운로드 받기
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#데이터셋 불러오기, 중복 제거
df = pd.read_csv('./datasets/filtered_reviews_3000_per_category.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# print(df.head())
# print(df.info())


#장르 값 체크
print(df.genre.value_counts())

X = df['review']
Y = df['genre']

# 입력값 출력값 첫 데이터 체크
print("First Title:", X[0])
print("First genre:", Y[0])

# lemmatizer(형태소 분류기 / 표제어 추출기) 토큰화, stopwords정의
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

#토크나이저 함수 정의
def preprocess_text(sentence):
    tokens = word_tokenize(sentence)  # Tokenize into words
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Keep only alphabetic words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize words
    return tokens

# 토크나이저 테스트
processed_x = preprocess_text(X[0])
print("Processed Tokens:", processed_x)

# Encode categories into numerical labels
encoder = LabelEncoder()

# Fit and transform the labels (only once to ensure consistency)
labeled_y = encoder.fit_transform(Y)
print("Encoded Labels:", labeled_y[:3])

# Save the label encoder for future use
with open('./models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

# Check the encoded label classes
label = encoder.classes_
print("Label Classes:", label)

# Apply preprocessing to all titles
X = X.apply(lambda x: ' '.join(preprocess_text(x)))
print("Processed Titles:", X[:5])

# 토크나이저 생성하기 (토큰이 너무 많기 때문에 10000으로 제한)
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X)

tokenizer = Tokenizer(num_words=10000)  # Limit vocabulary to top 10,000 words
tokenizer.fit_on_texts(X)


# 생성된 토크나이저로 전체 데이터 토큰화 하기
tokenized_X = tokenizer.texts_to_sequences(X)

# 총 형태소 수 구하기
word_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:", word_size)

# 토크나이저 저장
with open('./models/tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)

# Pad sequences to ensure uniform input size
max_length = max(len(seq) for seq in tokenized_X)  # Find the maximum sequence length
X_pad = pad_sequences(tokenized_X, maxlen=max_length, padding='post')  # Pad with zeros at the end

# 출력 라벨 One-hot 엔코딩하기
onehot_Y = to_categorical(labeled_y)

# 데이터 세트 학습용과 테스트용으로 나누기
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)
print("Training Data Shape:", X_train.shape, Y_train.shape)
print("Testing Data Shape:", X_test.shape, Y_test.shape)

# Save processed data
np.save('datasets/filtered_reviews_X_train.npy', X_train)
np.save('datasets/filtered_reviews_Y_train.npy', Y_train)
np.save('datasets/filtered_reviews_X_test.npy', X_test)
np.save('datasets/filtered_reviews_Y_test.npy', Y_test)

# exit()
#
# okt = Okt()
# okt_x = okt.morphs(X[0])
#
#
#
# encoder = LabelEncoder()
#
# #fit transform은 한번만 해야하고,  encoder는 저장을 해놓고, 그다음 추가 데이터가 구해지면 이 엔코더를 새로만드는것이 아니라 그대로 사용해야 한다
# labeled_y = encoder.fit_transform(Y)
# print(labeled_y[:3])
#
# label = encoder.classes_
# print(label)
#
# #pickle은 텍스트 형태로 저장하는 것이 아니라 파일의 형태를 그대로 유지한다(바이너리 코드 형태로), 그래서 따로 읽어보는것은 불가하다
# #wb = binary write
# #with문은 with가 끝나면 닫는 명령어 (while문과 비슷)
# with open('./models/encoder.pickle', 'wb') as f:
#     pickle.dump(encoder, f)
#
# #토큰화 -> 형태소 단위로 모든 단어들을 나누는것
# #원형화 -> 모든 형태소를 원형으로 바꾸는 것
#
# onehot_Y = to_categorical(labeled_y)
# print(onehot_Y)
#
# for i in range(len(X)):
#     X[i] = okt.morphs(X[i], stem=True)
# print(X)
#
#
# #불용어 제거
# stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col = 0)
# print(stopwords)
#
# # 문장 수만큼 for문 돌리기
# for sentence in range(len(X)):
#
#     #사용할 단어 리스트 선언
#     words = []
#
#     #한문장의 형태소 수만큼 두번째 for문 돌리기
#     for word in range(len(X[sentence])):
#         #단어의 길이가 2이상일때만
#         if len(X[sentence][word]) > 1:
#             #stopwords에 있는 모든 단어 다 빼기, 1글자짜리 형태소 다빼기
#             if X[sentence][word] not in list(stopwords['stopword']):
#                 words.append(X[sentence][word])
#
#     # 다시 문장으로 조합하기
#     X[sentence] = ' '.join(words)
#
# print('X ====', X[:5])
#
# token = Tokenizer()
#
# #text를 기반으로한 tokenizer 리스트 만들기
# token.fit_on_texts(X)
#
# #tokenizer 리스트를 문자구조로 만들기
# tokened_X = token.texts_to_sequences(X)
#
# #0을 사용해야하기 때문에 (빈 단어) 형태소 총 개수를 1을 더해줌
# wordsize = len(token.word_index) + 1
#
# print(wordsize)
#
#
#
# # 최대 문장길이 찾는 알고리즘
# max = 0
# for i in range(len(tokened_X)):
#     if max < len(tokened_X[i]):
#         max = len(tokened_X[i])
#
# print('max words:', max)
#
# #토큰 저장 (테스트에 사용해야함)
# with open('./models/news_token_max_{}.pickle'.format(max), 'wb') as f:
#     pickle.dump(token, f)
#
#
# print(tokened_X[0])
#
#
# #문장마다 길이가 다르기 때문에 가장 긴 문장으로 맞추고 나머지 문장들의 빈단어 대신 0으로 채우되 0을 앞에다 채움
# #텐서플로우 함수 pad_sequences를 사용해서 0으로 채우기
# X_pad = pad_sequences(tokened_X, max)
# print(X_pad)
# print(len(X_pad[0]))
#
# X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size = 0.1)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)
#
# np.save('./crawling_data/news_data_X_train_wordsize{}_max_{}'.format(wordsize, max), X_train)
# np.save('./crawling_data/news_data_Y_train_wordsize{}_max_{}'.format(wordsize, max), Y_train)
# np.save('./crawling_data/news_data_X_test_wordsize{}_max_{}'.format(wordsize, max), X_test)
# np.save('./crawling_data/news_data_Y_test_wordsize{}_max_{}'.format(wordsize, max), Y_test)
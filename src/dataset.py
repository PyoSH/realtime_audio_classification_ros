import copy
import glob

import pandas as pd
from torch.utils.data import Dataset
from src.feature_extraction import *
from loguru import logger
from tqdm import tqdm

class AudioDataset(Dataset):
    def __init__(self, X, y, input_size):
        #  목표 입력 텐서는 batch size 128 for 50ms , frame size = 2205, input_size = 40 (128, 2205??, 40)
        self.X = torch.tensor(X[:, :, :input_size], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) # long -> float32 -> long

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
class ProtoDataset(Dataset):
    def __init__(self, dataPath, featureParams, item=1):
        self.audioPath = os.path.join(dataPath,'audio',f'TestSet_{item}_1.mp3')
        self.labelPath = os.path.join(dataPath,'label', f'test{item}.txt')
        self.featureParams = featureParams

        self.featureVector = get_mp3_to_mfcc(self.audioPath, self.featureParams.sr, self.featureParams.n_mfcc,
                                             self.featureParams.hop_length, self.featureParams.len_fft).T # 전치!!!
        self.label_raw = pd.read_csv(self.labelPath, header=None, sep='\t')
        self.label_processed = None

    def __len__(self):
        return len(self.featureVector)

    def __getitem__(self, idx):
        return self.featureVector[idx], self.label_processed[idx]

'''
labelProcessing 2024-04-04 표승현
예측 수행할 때 라밸 데이터를 특징 벡터에 맞게 나눠주는 함수
dataset_bringup에서 사용
'''
def labelProcessing(label_raw, vectorShape, label_class, sampleRate, len_frame):
    label_raw.columns = ['start', 'end', 'label']

    # 레이블을 숫자로 매핑
    label_mapping = array_to_dict(label_class)

    # 레이블 데이터 준비
    label_processed = np.zeros(vectorShape)  # 모델 입력 차원(= 특징 벡터 열 개수)에 맞는 레이블 배열 초기화
    for _, row in label_raw.iterrows():
        start_frame = int(row['start'] * sampleRate / len_frame) # audio frame 단위.
        end_frame = int(row['end'] * sampleRate / len_frame)
        temp_label = None
        if row['label'] == 'hardcutting_v' or row['label'] == 'hardcutting_h': temp_label = 'hardcutting'
        else: temp_label = row['label']
        # label_processed[start_frame:end_frame] = label_mapping[row['label']]
        label_processed[start_frame:end_frame] = label_mapping[temp_label]

    return label_processed

'''
audioProcessing 2024-04-04 표승현
dataset_bringup으로 처리한 행렬 중 음향 데이터를 mfcc로 특징 추출하는 함수
'''
def audioProcessing(data, mfcc_const):
    data_featureVector = None
    a = range(data.shape[0])
    for idx in tqdm(a, desc="Feature extraction progressing"):
        temp_data = data[idx, :]
        temp_featureVector = get_frame_to_mfcc(temp_data, samplingRate=mfcc_const.sr,
                                               num_cepstralCoefficient=mfcc_const.n_mfcc,
                                               hop_length=mfcc_const.hop_length, len_fft=mfcc_const.len_fft)
        if idx == 0:
            data_featureVector = copy.deepcopy(temp_featureVector)
        else:
            data_featureVector = np.append(data_featureVector, temp_featureVector, axis=0)

    return data_featureVector

def zeropad1d(A, length):
    retVal = np.zeros(length)
    retVal[:len(A)] = A
    return retVal

def array_to_dict(arr):
    return {arr[i] : i for i in range(len(arr))}

def file_path(path, fileName):
    return os.path.join(path, '{}'.format(fileName))
def is_audio(fileName):
    EXTENSIONS = ['.mp3', '.wav']
    return any(fileName.endswith(ext) for ext in EXTENSIONS)

def list_audio_files(dir):
    EXTENSIONS = ['*.mp3', '*.wav']
    audio_list = []
    for EXTENSION in EXTENSIONS:
        audio_list.extend(glob.glob(os.path.join(dir, '**', EXTENSION), recursive=True))

    audio_list.sort(key=lambda path: os.path.basename(path))

    return audio_list

def list_label_files(dir):
    EXTENSIONS = ['*.txt']
    label_list = []
    for EXTENSION in EXTENSIONS:
        label_list.extend(glob.glob(os.path.join(dir, '**', EXTENSION), recursive=True))

    label_list.sort(key=lambda path: os.path.basename(path))

    return label_list

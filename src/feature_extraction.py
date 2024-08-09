import os
import librosa
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
import copy

class MFCC_params():
    def __init__(self, samplingRate, num_CepstralCs, hop_length, len_window):
        self.sr = samplingRate
        self.n_mfcc = num_CepstralCs
        self.hop_length = hop_length # 2의 배수일 것, hop_length = n_fft /4
        self.len_fft = len_window # n_fft 의미

def save_mp3_to_mfcc(mp3_dir, save_dir, sr, n_mfcc, hop_length, len_fft):
    try:
        # Load audio file using librosa with specified parameters
        audio_data, _ = librosa.load(mp3_dir, sr=sr)

        # Extract MFCC features with specified parameters
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft = len_fft)

        # Normalize MFCC data using StandardScaler
        scaler = StandardScaler()
        mfccs_normalized = scaler.fit_transform(mfccs)

        # Save normalized MFCC data as a NumPy file
        mfcc_normalized_file_path = os.path.join(save_dir, os.path.basename(mp3_dir).replace('.mp3', f'_mfcc_normalized.npy'))
        np.save(mfcc_normalized_file_path, mfccs_normalized)

        print(f"Conversion successful: {mp3_dir} -> {mfcc_normalized_file_path}")
    except Exception as e:
        print(f"Error processing {mp3_dir}: {e}")
'''
유진님 코드에서 가져와 구현해놨고, 이전 ProtoDataset에서 사용했으나
frame_to_mfcc가 seq_len 조절이 가능해서 안씀
표승현 2024-04-04 
'''
def get_mp3_to_mfcc(mp3_dir, sr, n_mfcc, hop_length, len_fft):
    try:
        # Load audio file using librosa with specified parameters
        audio_data, _ = librosa.load(mp3_dir, sr=sr)

        # Extract MFCC features with specified parameters
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft = len_fft)

        # Normalize MFCC data using StandardScaler
        scaler = StandardScaler()
        mfccs_normalized = scaler.fit_transform(mfccs)

        return mfccs_normalized

    except Exception as e:
        print(f"Error processing {mp3_dir}: {e}")

'''
< for real time, data means frame!!>
leng_fft = n_fft
num_cepstralCoefficient = n_mfcc
'''
def get_frame_to_mfcc(data, samplingRate, num_cepstralCoefficient, hop_length, len_fft):
    # 1-1 프레임 대한 정규화 (-1~1) (before mfcc)
    # 16-bit 정수를 부동 소수점으로 변환
    frame_float = data.astype(np.float32) / 32767.0
    frame_norm = None
    # 1-2 프레임 대한 표준화
    if np.max(frame_float) == 0:
        frame_norm = frame_float
    else:
        frame_norm = (frame_float - np.mean(frame_float)) / np.std(frame_float)

    # 2 프레임 정규화 X
    # frame_norm = copy.deepcopy(data.astype(np.float32))

    # 특징 벡터 추출
    featureVector = librosa.feature.mfcc(y=frame_norm, sr=samplingRate, hop_length=hop_length, n_mfcc=num_cepstralCoefficient,
                                 n_fft=len_fft).T # 전치!!!

    # 3 프레임 정규화 (0~1)
    # featureVector = minmax_scale(copy.deepcopy(featureVector), feature_range=(0, 1), axis=0)
    #차원 추가!!!
    featureVector = featureVector[np.newaxis, :, :]
    return featureVector

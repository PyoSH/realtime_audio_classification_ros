import sys
import os
[sys.path.append(i) for i in ['.', '..']]
#sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import rospy
from audio_common_msgs.msg import AudioData
from src.feature_extraction import *
from src.model_definition import *
from src.dataset import *
from src.audio_utils import *
from config import cfg, update_config
import numpy as np
import argparse
from loguru import logger
import torch
from pydub import AudioSegment
import io
import ffmpeg

parser = argparse.ArgumentParser(description='Running audio classification with ROS')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)

args = parser.parse_args()
update_config(cfg, args)

# MFCC 설정
mfcc_const = MFCC_params(cfg.FEATUREPARAMS.SAMPLING_RATE, cfg.FEATUREPARAMS.NUM_CEPSTRAL_COEFFICIENTS,
                         cfg.FEATUREPARAMS.HOP_LENGTH, cfg.FEATUREPARAMS.LEN_WINDOW)
CHUNK = cfg.FEATUREPARAMS.CHUNK
class_labels = cfg.HYPERPARAMS.LABEL_CLASS

# 모델 초기화
model = None
if cfg.HYPERPARAMS.MODELTYPE == 'RNN':
    model = RNNModel(input_dim=mfcc_const.n_mfcc, hidden_dim=cfg.HYPERPARAMS.HIDDEN_SIZE,
                     num_layers=cfg.HYPERPARAMS.NUM_LAYERS,
                     output_dim=cfg.HYPERPARAMS.NUM_CLASSES)
elif cfg.HYPERPARAMS.MODELTYPE == 'LSTM':
    model = LSTMModel(input_dim=mfcc_const.n_mfcc, hidden_dim=cfg.HYPERPARAMS.HIDDEN_SIZE,
                      num_layers=cfg.HYPERPARAMS.NUM_LAYERS,
                      output_dim=cfg.HYPERPARAMS.NUM_CLASSES)
model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH))

# ROS 노드 초기화
rospy.init_node('audio_classification_node', anonymous=True)
logger.info("ROS Audio Classification Node Started")

def audio_callback(data):    
    process = (
            ffmpeg
            .input('pipe:0', format='mp3', fflags='nobuffer', **{'re': None})
            .output('pipe:1', format='s16le', acodec='pcm_s16le', ac=1, ar=44100)
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            )
    out, err = process.communicate(input=bytes(data.data))
    logger.info(err.decode('utf-8'))
    input_np = np.frombuffer(out, dtype=np.int16)
    
    # 특징 추출
    featureVector = get_frame_to_mfcc(input_np, samplingRate=mfcc_const.sr, num_cepstralCoefficient=mfcc_const.n_mfcc,
                                      hop_length=mfcc_const.hop_length, len_fft=mfcc_const.len_fft)
    input_tensor = torch.tensor(featureVector[:, :, :mfcc_const.n_mfcc], dtype=torch.float32)
    
    # 모델 예측
    with torch.no_grad():
        model.eval()
        outputs = model(input_tensor)
    
    _, predicted = torch.max(outputs.data, 1)
    predicted_class = class_labels[predicted.item()]
    logger.info(f'Predicted: {predicted_class}')
    logger.info(f'length of data: {len(data)}')

def listener():
    # 오디오 데이터 토픽 구독
    rospy.Subscriber("/audio/audio_2", AudioData, audio_callback)
    
    # ROS 스핀 실행 (콜백이 계속 동작하도록)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        logger.info("ROS Audio Classification Node Stopped")
        pass


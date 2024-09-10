import sys
import os
[sys.path.append(i) for i in ['.', '..']]

import rospy
from audio_common_msgs.msg import AudioData
import miniaudio
from loguru import logger
import numpy as np
import argparse
from src.feature_extraction import *
from src.model_definition import *
from src.dataset import *
from src.audio_utils import *
from config import cfg, update_config
from torchsummary import summary

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

logger.info(f"model {model}")

class AudioStreamDecoder:
    def __init__(self):
        logger.info("decodeing started...")
        self.audio_stream = b''  # MP3 데이터를 버퍼링할 변수
        self.chunk_size = 1024   # MP3 데이터를 일정 크기로 자를 때 사용

        # ROS에서 audioData 토픽 구독
        rospy.init_node('audio_stream_decoder', anonymous=True)
        rospy.Subscriber("/audio/audio_2", AudioData, self.audio_callback)

    def audio_callback(self, msg):
        # audioData 메시지에서 MP3 바이트 데이터를 수신
        self.audio_stream += bytes(msg.data)
        # logger.info(f"current mp3 data {len(msg.data)} total mp3 data {len(self.audio_stream)}")

        # 일정 크기의 데이터가 쌓이면 디코딩
        if len(self.audio_stream) > self.chunk_size:
            self.decode_mp3_stream()

    def decode_mp3_stream(self):
        try:
            # 누적된 MP3 데이터를 miniaudio로 PCM으로 디코딩
            pcm_data = miniaudio.decode(self.audio_stream)

            # PCM 프레임 수 출력
            curr_pcm = np.array(pcm_data.samples)
            # sliced_pcm = curr_pcm[:4410]
            # del(curr_pcm)
            
            # 받은 MP3 데이터를 처리한 후 초기화 (혹은 사용한 만큼만 잘라내기)
            self.audio_stream = b''

            self.predict_situation(curr_pcm)
        
        except Exception as e:
            rospy.logerr(f"Error decoding MP3 stream: {e}")

    def predict_situation(self, input_np):
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
        logger.info(f'Predicted: {predicted_class}, {predicted.item()}')


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        decoder = AudioStreamDecoder()
        decoder.run()
    except rospy.ROSInterruptException:
        pass

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

class RealTimePredictor:
    def __init__(self, dtype=None):
        logger.info("started...")
        self.audio_stream = b''
        self.data_format = dtype
        self.mp3_chunk_size = 1024   # MP3 데이터를 일정 크기로 자를 때 사용
        self.pcm_chunk_size = 4410  # PCM 데이터를 일정 크기로 자를 때 사용

        rospy.init_node('realTimePredictor', anonymous=True)
        rospy.Subscriber("/audio/audio_2", AudioData, self.audio_callback)

    def audio_callback(self, msg):
        # audioData 메시지에서 MP3 바이트 데이터를 수신
        # self.audio_stream += bytes(msg.data)
        # logger.info(
        #     f"current {self.data_format} data {len(msg.data)} total {self.data_format} data {len(self.audio_stream)}")

        if (self.data_format == 'mp3') and (len(self.audio_stream) >= self.mp3_chunk_size):
            self.decode_stream()
        elif (self.data_format == 'wave') and(len(self.audio_stream) >= self.pcm_chunk_size):
            self.decode_stream()
        else:
            self.audio_stream += bytes(msg.data)

    def decode_stream(self):
        try:
            curr_pcm = None
            if self.data_format == 'mp3':
                # 누적된 MP3 데이터를 miniaudio로 PCM으로 디코딩
                pcm_data = miniaudio.decode(self.audio_stream)
                curr_pcm = np.array(pcm_data.samples)
            elif self.data_format == 'wave':
                curr_pcm = np.frombuffer(self.audio_stream, dtype=np.int16)
            else:
                logger.error(f"{self.data_format} format is not usable")
            # logger.info(f"current {self.data_format} data {len(self.audio_stream)}")
            sliced_pcm = curr_pcm[:4410]
            del(curr_pcm)

            self.predict_situation(sliced_pcm)
            self.audio_stream = b''

        except Exception as e:
            rospy.logerr(f"Error decoding MP3 stream: {e}")

    def predict_situation(self, input_np):
        # 특징 추출
        feature_vector = get_frame_to_mfcc(input_np, samplingRate=mfcc_const.sr, num_cepstralCoefficient=mfcc_const.n_mfcc,
                                        hop_length=mfcc_const.hop_length, len_fft=mfcc_const.len_fft)
        input_tensor = torch.tensor(feature_vector[:, :, :mfcc_const.n_mfcc], dtype=torch.float32)
        
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
        data_type = str(rospy.get_param('/audio/audio_capture/format'))
        predictor = RealTimePredictor(data_type)
        predictor.run()
    except rospy.ROSInterruptException:
        pass

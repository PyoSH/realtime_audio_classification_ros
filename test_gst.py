import sys
import os
[sys.path.append(i) for i in ['.', '..']]

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

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import threading

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

class RosGstDecode:
    def __init__(self, audio_consts_, model_):
        Gst.init(None)
        
        self.pred_model = model_
        self.audio_consts = audio_consts_
        
        self.channels = 1
        self.depth = 16
        self.sample_rate = audio_consts_.sr
        self.sample_format = str("S16LE")
        self.data_accum = None

        self.pipeline = Gst.Pipeline.new("app_pipeline")
        self.source = Gst.ElementFactory.make("appsrc", "app_source")
        self.decoder = Gst.ElementFactory.make("decodebin", "decoder")
        self.filter = Gst.ElementFactory.make("capsfilter", "filter")
        self.appsink = Gst.ElementFactory.make("appsink", "app_sink")
        caps = Gst.Caps.from_string(
            "audio/x-raw, format=(string)S16LE, channels=(int)1, rate=(int)44100, layout=(string)interleaved")
        
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.set_property("max-buffers", 2)
        self.appsink.set_property("drop", True)
        self.source.set_property("do-timestamp", True)
        self.source.set_property("caps", caps)

        self.pipeline.add(self.source)
        self.pipeline.add(self.decoder)
        self.pipeline.add(self.filter)
        self.pipeline.add(self.appsink)
        
        self.filter.link(self.appsink)

        self.decoder.connect("pad-added", self.cb_newpad)
        self.appsink.connect("new-sample", self.on_new_sample)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Unable to set the pipeline to the playing state")
            exit(-1)
        
        self.loop = GLib.MainLoop()
        self.thread = threading.Thread(target=self.loop.run)
        self.thread.start()
        
        logger.info("GStreamer 파이프라인이 실행되었습니다.")
        
        self.subscriber = rospy.Subscriber("/audio/audio_2", AudioData, self.on_audio)

    def on_audio(self, msg):
        # logger.info("Received audio message")
        logger.info(self.pipeline.get_state(Gst.CLOCK_TIME_NONE)[1])
        # buf = Gst.Buffer.new_allocate(None, len(msg.data), None)
        # buf.fill(0, msg.data)        
        # self.source.emit("push-buffer", buf)
        
        # 파이프라인 상태 확인
        currState = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)[1]
        if currState == Gst.State.PLAYING:
            buf = Gst.Buffer.new_allocate(None, len(msg.data), None)
            buf.fill(0, msg.data)
            self.source.emit("push-buffer", buf)
        elif currState == Gst.State.READY:
            logger.info("Attempting to push buffer before pipeline is in PLAYING state")
            
            buf = Gst.Buffer.new_allocate(None, len(msg.data), None)
            buf.fill(0, msg.data)
            self.source.emit("push-buffer", buf)
        else:
            pass
        


    def cb_newpad(self, decodebin, pad):
        logger.info(f"New pad added: {pad.get_name()}")
        if pad.is_linked():
            logger.info("Pad is already linked")
            return
        sink_pad = self.filter.get_static_pad("sink")
        if sink_pad.is_linked():
            logger.info("Sink pad is already linked")
            return
        result = pad.link(sink_pad)
        if result == Gst.PadLinkReturn.OK:
            logger.info(f"Linked {pad.get_name()} to {sink_pad.get_name()}")
        else:
            logger.error(f"Failed to link pad: {result}")


    def on_new_sample(self, appsink):
        
        sample = appsink.emit("pull-sample")
        buf = sample.get_buffer()
        data = buf.extract_dup(0, buf.get_size())
        
        if len(data) %2 != 0:
            data = data[:-1]
        
        if self.data_accum == None:
            self.data_accum = np.frombuffer(data, dtype=np.int16)
        
        else:
            if len(self.data_accum) < 4410:
                data_temp = np.frombuffer(data, dtype=np.int16)
                self.data_accum = np.hstack((self.data_accum, data_temp))
                
                logger.info(f'current len: {len(data)}, accumulated len: {len(self.data_accum)}')
                
            else:                
                featureVector = get_frame_to_mfcc(self.data_accum, samplingRate=self.audio_consts.sr, num_cepstralCoefficient=self.audio_consts.n_mfcc,
                                                  hop_length=self.audio_consts.hop_length, len_fft=self.audio_consts.len_fft)
                input_tensor = torch.tensor(featureVector[:, :, :self.audio_consts.n_mfcc], dtype=torch.float32)
                self.data_accum = None
                
                with torch.no_grad():
                    self.pred_model.eval()
                    outputs = self.pred_model(input_tensor)
                
                _, predicted = torch.max(outputs.data, 1)
                predicted_class = class_labels[predicted.item()]
                logger.info(f'Predicted: {predicted_class}')
        return Gst.FlowReturn.OK

    def shutdown(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()
        self.thread.join()
        
if __name__ == '__main__':
    rospy.init_node('audio_decode')
    decoder = RosGstDecode(mfcc_const, model)
    rospy.spin()
    decoder.shutdown()

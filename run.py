import os

import numpy as np
from inaSpeechSegmenter import tf_mfcc

from formants import predict_from_times

if __name__ == '__main__':
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/cuda'
    # predict_from_times('data/VT 150hz baseline example.mp3', 'data/VT Predictions.csv', 0, 1)
    # tf_mfcc.power_spectrum(np.zeros(1024, dtype=np.int16), 1024, 512)
    predict_from_times('data/Example-f32le.wav', 'data/Example-F32-Predictions.csv', 0, 1)
    # predict_from_times('data/Example.wav', 'data/Example-Predictions.csv', 0, 1)

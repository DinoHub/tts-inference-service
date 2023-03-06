from typing import Union, Tuple

import torch
import numpy as np
import gradio as gr
import gradio.processing_utils as gr_processing_utils
import soundfile as sf
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.tts.models import FastPitchModel

audio_postprocess_ori = gr.Audio.postprocess

def audio_postprocess(self, y):
    data = audio_postprocess_ori(self, y)
    if data is None:
        return None
    return gr_processing_utils.encode_url_or_file_to_base64(data["name"])

gr.Audio.postprocess = audio_postprocess

from config import config, BaseConfig

''' CPU/GPU Configurations '''
if torch.cuda.is_available():
    DEVICE = [0]  # use 0th CUDA device
    ACCELERATOR = 'gpu'
else:
    DEVICE = 1
    ACCELERATOR = 'cpu'

MAP_LOCATION: str = torch.device('cuda:{}'.format(DEVICE[0]) if ACCELERATOR == 'gpu' else 'cpu')

''' Gradio Input/Output Configurations '''
inputs: str = 'text'
# inputs: Union[str, gr.inputs.Audio] = gr.inputs.Audio(source='upload', type='filepath')
outputs: gr.Audio = gr.Audio()

''' Helper functions '''
def initialize_tts_models(cfg: BaseConfig) -> str:

    spec_model = FastPitchModel.restore_from(restore_path=cfg.spec_model_path)
    spec_model.eval()

    vocoder_model = HifiGanModel.restore_from(restore_path=cfg.vocoder_model_path)
    vocoder_model = vocoder_model.eval()

    if torch.cuda.is_available():
        spec_model.cuda()
        vocoder_model.cuda()

    return spec_model, vocoder_model

''' Initialize models '''
spec_model, vocoder_model = initialize_tts_models(config)

''' Main prediction function '''
def predict(text: str) -> Tuple[Union[str, np.ndarray]]:

    speaker = None

    with torch.no_grad():
        parsed = spec_model.parse(text)
        if speaker is not None:
            speaker = torch.tensor([speaker]).long().to(device=spec_model.device)
        spectrogram = spec_model.generate_spectrogram(tokens=parsed, speaker=speaker)
        audio = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)

    if len(audio.shape) == 2:
        audio = audio.squeeze(0)

    if isinstance(audio, torch.Tensor):
        audio = audio.to('cpu').numpy()
    
    return (config.sample_rate, audio)

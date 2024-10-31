# 完美运行
# 只有通过提示音频克隆的功能
# 没有保存音色的功能

import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1" 

# 仅保留'3s极速复刻'模式
inference_mode_list = ['3s极速复刻']
instruct_dict = {'3s极速复刻': '1. 选择或录制提示音频，注意不超过30s\n2. 输入提示文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def generate_audio(tts_text, prompt_text, prompt_wav_upload, prompt_wav_record,
                   seed, stream, speed):
    
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    if prompt_wav is None:
        gr.Warning('提示音频为空，请提供提示音频。')
        yield (target_sr, default_data)

    if prompt_text == '':
        gr.Warning('提示文本为空，请输入提示文本。')
        yield (target_sr, default_data)

    if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
        gr.Warning('提示音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
        yield (target_sr, default_data)

    logging.info('开始克隆声音并合成语音...')
    prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
    set_all_random_seed(seed)

    if stream:
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    else:
        tts_speeches = []
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            tts_speeches.append(i['tts_speech'])
        audio_data = torch.concat(tts_speeches, dim=1)
        yield (target_sr, audio_data.numpy().flatten())

def main():
    with gr.Blocks() as demo:
        gr.Markdown("<center><span style='font-size: 54px;'>声音克隆与语音合成</span></center>")
        gr.Markdown("<center><span style='font-size: 18px;'>请输入需要合成的文本，上传或录制提示音频，并输入提示文本</span></center>")
        
        tts_text = gr.Textbox(label="输入合成文本", lines=3, value="请输入您要合成的文本内容。")
        prompt_text = gr.Textbox(label="输入提示文本", lines=3, value="请输入提示音频对应的文本内容。")
        seed_button = gr.Button(value="\U0001F3B2")
        seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='上传提示音频（不超过30秒）')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制提示音频（不超过30秒）')

        speed = gr.Number(value=1.0, label="语速调整（仅非流式推理）", minimum=0.5, maximum=2.0, step=0.1)
        stream = gr.Checkbox(label="是否流式推理", value=False)
        generate_button = gr.Button("生成音频")
        audio_output = gr.Audio(label="合成音频", autoplay=False, streaming=False, show_download_button=True)
        
        seed_button.click(generate_seed, inputs=[], outputs=seed)
        
        generate_button.click(generate_audio,
                              inputs=[tts_text, prompt_text, prompt_wav_upload, prompt_wav_record,
                                      seed, stream, speed],
                              outputs=[audio_output])
        
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port, inbrowser=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=7993)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='模型路径或 ModelScope 仓库 ID')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()

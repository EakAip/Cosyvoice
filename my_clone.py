# 主程序测试demo

# 声音克隆 保存音色特征

# 使用保存的音色进行推理


import os
import re
import sys
import time
import shutil
import numpy as np
import torch
import torchaudio
import random
import librosa
import argparse
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed
from gradio_client import Client, file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))


# 获取已保存的音色列表
def get_saved_voices():
    voices_dir = os.path.join(ROOT_DIR, 'voices')
    if not os.path.exists(voices_dir):
        os.makedirs(voices_dir)
    voices = [f.replace('.pt', '') for f in os.listdir(voices_dir) if f.endswith('.pt')]
    return voices

# 优化生成音频质量 是其更自然
def postprocess(speech, top_db=60, hop_length=220, win_length=440): # 这段代码的目的是优化语音信号的质量，去除不必要的静音、避免失真，并为输出添加适当的静音，以确保更自然的声音效果。
    max_val = 0.8
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


def clone_voice(prompt_wav_upload, spk_name,seed=42, speed=1.0, tts_text="你好,我是你的数字人"):

    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    else:
        return '提示音频为空，请提供提示音频。'
    
    client = Client("http://188.18.18.106:7998/")
    result = client.predict(audio_file=file(prompt_wav_upload),hotwords=" ",api_name="/recognize_audio")
    prompt_text = re.sub(r'.*speaker0:', '', result[0])         # 使用正则，result只保留speaker0:之后的内容。
    print(prompt_text)

    if prompt_text == '':
        return '提示文本为空，请输入提示文本。'

    if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
        return '提示音频采样率过低，请提供采样率不低于16kHz的音频'

    logging.info('开始克隆声音并合成语音...')
    prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
    set_all_random_seed(seed)

    # 克隆声音并保存音色
    tts_speeches = []
    for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, spk_name=spk_name,stream=False, speed=speed):
        tts_speeches.append(i['tts_speech'])
    # audio_data未保存
    audio_data = torch.concat(tts_speeches, dim=1)
    return '训练完成',(target_sr, audio_data.numpy().flatten()) 



def infer(tts_text, spk_name, seed=42, speed=1.0):
    
    set_all_random_seed(seed)
    if not os.path.exists(os.path.join(ROOT_DIR, 'voices', f'{spk_name}.pt')):
        return '所选的音色文件不存在'
    # 使用已保存的音色进行推理
    tts_speeches = []
    for i in cosyvoice.inference_sft(tts_text, spk_id='中文女', stream=False, speed=speed, new_dropdown=spk_name):
        tts_speeches.append(i['tts_speech'])
    audio_data = torch.concat(tts_speeches, dim=1)
    return (target_sr, audio_data.numpy().flatten())



if __name__ == '__main__':
    # 模型路径 基本参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',type=str,default='pretrained_models/CosyVoice-300M',help='模型路径或 ModelScope 仓库 ID')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)


    # 克隆
    # prompt_wav_upload = "/opt/jyd01/wangruihua/api/digital/cosyvoice/参考音频/希望我们大家都能像他一样.wav"
    # spk_name = '含韵'
    
    # clone_voice(prompt_wav_upload, spk_name)


    # 推理
    tts_text="各位领导、各位老师，欢迎大家来到竞业达AIGC新质生产力中心。在这里，我们实现了从提供软件工具到直接提供基于AI结果服务的重大转变。首先，让我们谈谈我们的产品服务矩阵。经过一年多的努力，我们已成功构建了面向基础教育、高等教育和职业教育的全方位产品系列。这些服务不仅覆盖了课堂教学的各环节，还深入到了教师成长、学科质量提升以及教育管理的多个层面。特别是针对基础教育，我们聚焦于课堂质量分析、教师能力提升和薄弱学科改进，通过深度挖掘课堂数据，为学校和教师提供精准的改进建议。在高等教育和职业教育领域，我们的服务更加全面。我们不仅关注校内的教学数据，还广泛收集校外的产业需求信息，通过大数据分析，为高校提供从培养目标定位到教学质量评价，我们不仅关注校内的教学数据，还广泛收集校外的产业需求信息，通过大数据分析，为高校提供从培养目标定位到教学质量评价，我们不仅关注校内的教学数据，还广泛收集校外的产业需求信息，通过大数据分析，为高校提供从培养目标定位到教学质量评价，我们不仅关注校内的教学数据，还广泛收集校外的产业需求信息，通过大数据分析，为高校提供从培养目标定位到教学质量评价，我们不仅关注校内的教学。"
    spk_name = '聂'
    start_time = time.time()
    infer(tts_text, spk_name)
    end_time = time.time()
    print('推理时间：', end_time - start_time)

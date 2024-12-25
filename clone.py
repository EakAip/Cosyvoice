# 声音克隆 保存音色特征

# 使用保存的音色进行推理

# 专为接口设计，无缝对接原来接口

# 增加显存回收处理功能

# cosyvoice = CosyVoice(model_dir='pretrained_models/CosyVoice-300M')


import os
import re
import gc  # Python垃圾回收模块
import sys
import uuid
import shutil
import logging
import torch
import random
import librosa
import argparse
import torchaudio
import numpy as np
from flask import url_for
from flask import request
from cosyvoice.cli.cosyvoice2 import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed
from gradio_client import Client, file
from pydub import AudioSegment
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

SERVER_ADDRESS = "http://188.18.18.106"


# 获取已保存的音色列表
def get_saved_voices():
    voices_dir = os.path.join(ROOT_DIR, 'voices')
    if not os.path.exists(voices_dir):
        os.makedirs(voices_dir)
    voices = [f.replace('.pt', '') for f in os.listdir(voices_dir) if f.endswith('.pt')]
    return voices

# 优化生成音频质量 是其更自然
def postprocess(speech, top_db=60, hop_length=220, win_length=440): # 这段代码的目的是优化语音信号的质量，去除不必要的静音、避免失真，并为输出添加适当的静音，以确保更自然的声音效果。
    target_sr = 22050
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


def clone_voice(prompt_wav_upload, spk_name, seed=42, speed=1.0, tts_text="你好,我是你的数字人"):
    try:
        
        root_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(root_dir, 'pretrained_models', 'CosyVoice-300M')
        cosyvoice = CosyVoice(model_dir=model_path)
        
        
        # cosyvoice = CosyVoice(model_dir='pretrained_models/CosyVoice-300M')   # 相对路径 这里不要使用

        
        prompt_sr = 16000
        target_sr = 22050

        if prompt_wav_upload is not None:
            prompt_wav = prompt_wav_upload
        else:
            return '提示音频为空，请提供提示音频。', None

        client = Client("http://188.18.18.106:7998/")
        try:
            result = client.predict(audio_file=file(prompt_wav_upload), hotwords=" ", api_name="/recognize_audio")
        except Exception as e:
            print("提示音频识别失败，请提供可识别的音频。")
            return '提示音频识别失败，请提供可识别的音频。', None

        # 使用正则表达式过滤speaker后的内容
        prompt_text = re.sub(r'.*speaker\d+:\s*', '', result[0])
        print(prompt_text)

        if prompt_text == '':
            print("提示文本为空，请输入提示文本。")
            return '提示文本为空，请输入提示文本。', None

        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            print("提示音频采样率过低，请提供采样率不低于16kHz的音频")
            return '提示音频采样率过低，请提供采样率不低于16kHz的音频', None

        logging.info('开始克隆声音并合成语音...')
        print("开始克隆声音并合成语音...")
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)

        # 克隆声音并保存音色
        tts_speeches = []
        for i in cosyvoice.inference_zero_shot(
                tts_text, prompt_text, prompt_speech_16k, spk_name=spk_name, stream=False, speed=speed):
            tts_speeches.append(i['tts_speech'])

        # 合并生成的语音片段
        audio_data = torch.concat(tts_speeches, dim=1)

        # 清理内存和显存
        del cosyvoice, tts_speeches, prompt_speech_16k  # 删除不再使用的对象
        torch.cuda.empty_cache()  # 清理未使用的显存
        gc.collect()  # 进行垃圾回收
        print("训练完成")

        return '训练完成', (target_sr, audio_data.numpy().flatten())

    except Exception as e:
        logging.error("克隆过程中出错: %s", e)

        # 异常情况下也要释放资源
        torch.cuda.empty_cache()
        gc.collect()

        return str(e), None




def infer(tts_text, spk_name, seed=42, speed=0.9):
    try:
        if not os.path.exists(os.path.join(ROOT_DIR, 'voices', f'{spk_name}.pt')):
            return False


        cosyvoice = CosyVoice(model_dir=os.path.join(ROOT_DIR, 'pretrained_models/CosyVoice-300M'))

        target_sr = 22050
        set_all_random_seed(seed)

        # 使用已保存的音色进行推理
        tts_speeches = []
        for i in cosyvoice.inference_sft(tts_text, spk_id='中文女', stream=False, speed=speed, new_dropdown=spk_name):
            print(f"轮次：{i}")
            tts_speeches.append(i['tts_speech'])

        # 将所有生成的语音片段拼接在一起
        audio_data = torch.concat(tts_speeches, dim=1)

        # 确保 audio 目录存在
        audio_dir = os.path.join('static', 'audio')
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

        # 生成唯一的音频文件名
        output_filename = f'{uuid.uuid4()}.wav'
        output_path = os.path.join(audio_dir, output_filename)

        # 保存音频为 .wav 文件
        torchaudio.save(output_path, audio_data, target_sr)
        print(f"音频文件已保存到 {output_path}")

        # 释放显存
        del cosyvoice, tts_speeches, audio_data  # 删除对象以释放内存
        torch.cuda.empty_cache()  # 清理未使用的显存
        gc.collect()  # 强制进行垃圾回收


        base_url = SERVER_ADDRESS
        full_audio_url = base_url + "/sound_clone/" + output_path

        # print("*******************************")
        # 返回完整的音频文件 URL
        # base_url = request.url_root.rstrip('/')
        # full_audio_url = base_url + "/sound_clone/" + url_for('static', filename=os.path.join('audio', output_filename))
        
        # print(f"base_url:{base_url}")
        
        print(f"vioceid：{spk_name}推理音频路径:{full_audio_url}")
        
        
        return full_audio_url

    except Exception as e:
        print(f"推理错误: {e}")
        # 确保在发生错误时也释放资源
        torch.cuda.empty_cache()
        gc.collect()
        return False

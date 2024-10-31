# 克隆没问题 
# 音色推理没问题

# 刷新音色列表有问题


import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1" 

# 增加'预训练音色'模式
inference_mode_list = ['预训练音色', '3s极速复刻']
instruct_dict = {
    '预训练音色': '1. 选择已保存的音色\n2. 输入合成文本\n3. 点击生成音频按钮',
    '3s极速复刻': '1. 选择或录制提示音频，注意不超过30s\n2. 输入提示文本和合成文本\n3. 点击生成音频按钮\n4. 输入音色名称并保存克隆的音色（可选）'
}
max_val = 0.8

# 获取已保存的音色列表
def get_saved_voices():
    voices_dir = os.path.join(ROOT_DIR, 'voices')
    if not os.path.exists(voices_dir):
        os.makedirs(voices_dir)
    voices = [f.replace('.pt', '') for f in os.listdir(voices_dir) if f.endswith('.pt')]
    return voices

# 刷新音色列表
def refresh_voice_choices():
    voices = get_saved_voices()
    return gr.Dropdown.update(choices=voices)

# 保存克隆的音色
def save_voice(name):
    if not name or name.strip() == '':
        return gr.Alert("音色名称不能为空")
    shutil.copyfile(os.path.join(ROOT_DIR, 'output.pt'), os.path.join(ROOT_DIR, 'voices', f'{name}.pt'))
    return gr.Alert("音色保存成功")

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

def generate_audio(tts_text, mode_checkbox_group, saved_voice_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record,
                   seed, speed):
    
    if mode_checkbox_group == '预训练音色':
        if not saved_voice_dropdown:
            return gr.Alert('请选择一个已保存的音色')
        logging.info('使用已保存的音色进行语音合成...')
        set_all_random_seed(seed)
        if not os.path.exists(os.path.join(ROOT_DIR, 'voices', f'{saved_voice_dropdown}.pt')):
            return gr.Alert('所选的音色文件不存在')
        # 使用已保存的音色进行推理
        tts_speeches = []
        for i in cosyvoice.inference_sft(tts_text, spk_id='中文女', stream=False, speed=speed, new_dropdown=saved_voice_dropdown):
            tts_speeches.append(i['tts_speech'])
        audio_data = torch.concat(tts_speeches, dim=1)
        return (target_sr, audio_data.numpy().flatten())

    elif mode_checkbox_group == '3s极速复刻':
        if prompt_wav_upload is not None:
            prompt_wav = prompt_wav_upload
        elif prompt_wav_record is not None:
            prompt_wav = prompt_wav_record
        else:
            return gr.Alert('提示音频为空，请提供提示音频。')

        if prompt_text == '':
            return gr.Alert('提示文本为空，请输入提示文本。')

        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            return gr.Alert('提示音频采样率过低，请提供采样率不低于16kHz的音频')

        logging.info('开始克隆声音并合成语音...')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)

        # 克隆声音并保存音色
        tts_speeches = []
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False, speed=speed):
            tts_speeches.append(i['tts_speech'])
        audio_data = torch.concat(tts_speeches, dim=1)
        return (target_sr, audio_data.numpy().flatten())

def main():
    with gr.Blocks() as demo:
        gr.Markdown("<center><span style='font-size: 54px;'>声音克隆与语音合成</span></center>")
        gr.Markdown("<center><span style='font-size: 18px;'>请选择推理模式，并按照提示进行操作</span></center>")
        
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            instruction_text = gr.Textbox(label="操作步骤", value=instruct_dict[inference_mode_list[0]], interactive=False)
        
        tts_text = gr.Textbox(label="输入合成文本", lines=3, value="请输入您要合成的文本内容。")
        prompt_text = gr.Textbox(label="输入提示文本", lines=3, value="", placeholder="请输入提示音频对应的文本内容。")
        
        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='上传提示音频（不超过30秒）')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制提示音频（不超过30秒）')
        
        with gr.Row():
            # 定义刷新音色列表的按钮和下拉菜单
            saved_voice_dropdown = gr.Dropdown(label="选择已保存的音色", choices=get_saved_voices())
            refresh_button = gr.Button("刷新音色列表")

            # 绑定事件，注意 outputs 参数指向 saved_voice_dropdown
            refresh_button.click(fn=refresh_voice_choices, inputs=[], outputs=saved_voice_dropdown)

        
        seed_button = gr.Button(value="随机种子")
        seed = gr.Number(value=0, label="随机推理种子")
        seed_button.click(generate_seed, inputs=[], outputs=seed)
        
        speed = gr.Number(value=1.0, label="语速调整", minimum=0.5, maximum=2.0, step=0.1)
        
        generate_button = gr.Button("生成音频")
        audio_output = gr.Audio(label="合成音频", autoplay=False, streaming=False, show_download_button=True)
        
        # 保存音色部分
        new_voice_name = gr.Textbox(label="输入音色名称（用于保存克隆的音色）", lines=1, placeholder="输入音色名称")
        save_voice_button = gr.Button("保存克隆的音色")
        save_result = gr.Textbox(label="保存结果", interactive=False)
        # 保存音色部分
        def save_voice(name):
            if not name or name.strip() == '':
                return "音色名称不能为空"
            shutil.copyfile(os.path.join(ROOT_DIR, 'output.pt'), os.path.join(ROOT_DIR, 'voices', f'{name}.pt'))
            return "音色保存成功"

        save_voice_button.click(fn=save_voice, inputs=new_voice_name, outputs=save_result)

        
        # 根据模式显示或隐藏组件
        def update_visibility(mode):
            if mode == '预训练音色':
                return {
                    prompt_text: gr.update(visible=False),
                    prompt_wav_upload: gr.update(visible=False),
                    prompt_wav_record: gr.update(visible=False),
                    saved_voice_dropdown: gr.update(visible=True),
                    refresh_button: gr.update(visible=True),
                    new_voice_name: gr.update(visible=False),
                    save_voice_button: gr.update(visible=False),
                    save_result: gr.update(visible=False),
                }
            else:
                return {
                    prompt_text: gr.update(visible=True),
                    prompt_wav_upload: gr.update(visible=True),
                    prompt_wav_record: gr.update(visible=True),
                    saved_voice_dropdown: gr.update(visible=False),
                    refresh_button: gr.update(visible=False),
                    new_voice_name: gr.update(visible=True),
                    save_voice_button: gr.update(visible=True),
                    save_result: gr.update(visible=True),
                }
        
        mode_checkbox_group.change(fn=change_instruction, inputs=mode_checkbox_group, outputs=instruction_text)
        mode_checkbox_group.change(fn=update_visibility, inputs=mode_checkbox_group, outputs=[
            prompt_text, prompt_wav_upload, prompt_wav_record, saved_voice_dropdown, refresh_button,
            new_voice_name, save_voice_button, save_result
        ])
        
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, saved_voice_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record,
                                      seed, speed],
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

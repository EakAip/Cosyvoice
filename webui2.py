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

import shutil

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1" 



inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8




reference_wavs = ["请选择参考音频或者自己上传"]
for name in os.listdir(f"{ROOT_DIR}/参考音频/"):
    reference_wavs.append(name)

spk_new = ["无"]

for name in os.listdir(f"{ROOT_DIR}/voices/"):
    # print(name.replace(".pt",""))
    spk_new.append(name.replace(".pt",""))


def refresh_choices():

    spk_new = ["无"]

    for name in os.listdir(f"{ROOT_DIR}/voices/"):
        # print(name.replace(".pt",""))
        spk_new.append(name.replace(".pt",""))
    
    return {"choices":spk_new, "__type__": "update"}

def change_choices():

    reference_wavs = ["选择参考音频,或者自己上传"]

    for name in os.listdir(f"{ROOT_DIR}/参考音频/"):
        reference_wavs.append(name)
    
    return {"choices":reference_wavs, "__type__": "update"}


def change_wav(audio_path):

    text = audio_path.replace(".wav","").replace(".mp3","").replace(".WAV","")

    return f"{ROOT_DIR}/参考音频/{audio_path}",text


def save_name(name):

    if not name or name == "":
        gr.Info("音色名称不能为空")
        return False

    shutil.copyfile(f"{ROOT_DIR}/output.pt",f"{ROOT_DIR}/voices/{name}.pt")
    gr.Info("音色保存成功,存放位置为voices目录")

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


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed,new_dropdown):
    
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if cosyvoice.frontend.instruct is False:
            gr.Warning('您正在使用自然语言控制模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M-Instruct模型'.format(args.model_dir))
            yield (target_sr, default_data)
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            yield (target_sr, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if cosyvoice.frontend.instruct is True:
            gr.Warning('您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M模型'.format(args.model_dir))
            yield (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        if prompt_wav is None:
            gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
            yield (target_sr, default_data)
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            yield (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (target_sr, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            yield (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    if mode_checkbox_group == '预训练音色':
        logging.info('get sft inference request')
        set_all_random_seed(seed)

        if stream:
        
            for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed,new_dropdown=new_dropdown):
                yield (target_sr, i['tts_speech'].numpy().flatten())
        else:

            tts_speeches = []
            for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed,new_dropdown=new_dropdown):
                tts_speeches.append(i['tts_speech'])
            audio_data = torch.concat(tts_speeches, dim=1)
            yield (target_sr, audio_data.numpy().flatten())

            
    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
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
            
    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)

        if stream:
            
            for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
                yield (target_sr, i['tts_speech'].numpy().flatten())
        else:

            tts_speeches = []
            for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
                tts_speeches.append(i['tts_speech'])
            audio_data = torch.concat(tts_speeches, dim=1)
            yield (target_sr, audio_data.numpy().flatten())
            
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        if stream:
            
            for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed,new_dropdown=new_dropdown):
                yield (target_sr, i['tts_speech'].numpy().flatten())
        else:
            
            tts_speeches = []
            for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed,new_dropdown=new_dropdown):
                tts_speeches.append(i['tts_speech'])
            audio_data = torch.concat(tts_speeches, dim=1)
            yield (target_sr, audio_data.numpy().flatten())
            


def main():
    with gr.Blocks() as demo:
        gr.Markdown("<center><span style='font-size: 54px;'>声音克隆——JYD</span></center>")
        gr.Markdown("<center><span style='font-size: 54px;'> </span></center>")
        gr.Markdown("<center><span style='font-size: 18px;'>请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作</span></center>")
        gr.Markdown("<center><span style='font-size: 54px;'> </span></center>")
        
        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="栈和队列是两种重要的线性结构。从数据结构角度看，栈和队列也是线性表，其特殊性在于栈和队列的基本操作是线性表操作的子集，它们是操作受限的线性表，因此，可称为限定性的数据结构。")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[1])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0], scale=0.25)
            new_dropdown = gr.Dropdown(choices=spk_new, label='选择新增音色', value=spk_new[0],interactive=True)
            refresh_new_button = gr.Button("刷新新增音色")
            refresh_new_button.click(fn=refresh_choices, inputs=[], outputs=[new_dropdown])
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1],visible=False)
            stream_1 = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[1][1],visible=False)
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')

            with gr.Column():
                prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
                gr.Examples(
                [
                    "/opt/jyd01/wangruihua/api/digital/cosyvoice/参考音频/曹总原音.mp3",
                    "/opt/jyd01/wangruihua/api/digital/cosyvoice/参考音频/曹总_前9s.mp3",
                    "/opt/jyd01/wangruihua/api/digital/cosyvoice/参考音频/曹总_前17s.mp3",
                    "/opt/jyd01/wangruihua/api/digital/cosyvoice/参考音频/曹总9_17s.mp3",
                    "/opt/jyd01/wangruihua/api/digital/cosyvoice/参考音频/曹总17_28s.mp3",
                    
                ],
                inputs=[prompt_wav_upload]
                )
            wavs_dropdown = gr.Dropdown(label="参考音频列表",choices=reference_wavs,value="请选择参考音频或者自己上传",interactive=True)
            # refresh_button = gr.Button("刷新参考音频")
            # refresh_button.click(fn=change_choices, inputs=[], outputs=[wavs_dropdown])
            
            
        # prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='希望我们大家都能像他一样，不行他想了一下，我不能这样对国王说，这是在撒谎，但他们非常和气的问他说，你叫什么名字，鸭子心想，我必须去拿回我的软糖豆。')
        prompt_text = gr.Textbox(label="输入prompt文本（音频文件切分后，对应prompt文本也要相应减少）", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='同学们好! 今天我们继续讲《人工智能》这门课。这门课吧，说容易也容易，说难也有点难。 人工智能简称AI，是一个以计算机科学为基础，涉及多学科融合的交叉学科。今天这节课，我们重点讲讲深度学习和机器学习的基本原理，大家一定要注意，别混淆算法和模型。')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')

        new_name = gr.Textbox(label="输入新的音色名称", lines=1, placeholder="输入新的音色名称.", value='')

        save_button = gr.Button("保存刚刚推理的zero-shot音色")

        save_button.click(save_name, inputs=[new_name])

        wavs_dropdown.change(change_wav,[wavs_dropdown],[prompt_wav_upload,prompt_text])

        generate_button = gr.Button("生成音频",variant='primary')

        generate_button_stream = gr.Button("流式生成音频")

        audio_output = gr.Audio(label="合成音频", autoplay=False, streaming=False,show_download_button=True)

        audio_output_stream = gr.Audio(label="流式音频", autoplay=True, streaming=True,show_download_button=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed,new_dropdown],
                              outputs=[audio_output])

        generate_button_stream.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed,stream_1, speed,new_dropdown],
                              outputs=[audio_output_stream])
                              
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port,inbrowser=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=7995)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()


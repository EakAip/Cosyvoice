# 最后读完有个停顿 克隆出的效果最好

from flask import Flask, request, jsonify, copy_current_request_context
import threading
import os
from clone import clone_voice,infer # 导入 train 和 infer 函数
import queue
from concurrent.futures import ThreadPoolExecutor
import logging
from pydub import AudioSegment
import os

app = Flask(__name__)
inference_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=1)  # 确保同一时间只有一个推理任务在运行
lock = threading.Lock()

# 存储训练任务的状态
training_tasks = {}  # key: voiceid, value: {'status': code, 'remainder': seconds}


# 训练处理函数
def handle_training(voiceid, audio_file_path):
    
    with lock:
        training_tasks[voiceid] = {'status': 1, 'remainder': 180}  # 训练中状态

    try:
        train_status, _ = clone_voice(audio_file_path,voiceid)
        if train_status != '训练完成':
            with lock:
                training_tasks[voiceid] = {'status': 5, 'remainder': 0}  # 操作失败状态
            return

        with lock:
            training_tasks[voiceid] = {'status': 0, 'remainder': 0}  # 训练完成状态
    except Exception as e:
        logging.error("训练过程中出错: %s", e)
        with lock:
            training_tasks[voiceid] = {'status': 5, 'remainder': 0}  # 操作失败状态

# 接口1: 上传音频文件并开启声音训练任务
@app.route('/trainvoice', methods=['POST'])
def train_voice():
    voiceid = request.form.get('voiceid')
    print(f"收到训练请求，voiceid: {voiceid}")
    audio_file = request.files.get('voicefile')

    if not voiceid or not audio_file:
        return jsonify({"code": 5, "msg": "操作失败", "data": {}}), 400
    
    if audio_file.filename.endswith('.mp3'):
        # 定义 MP3 文件的保存路径
        audio_file_path = os.path.join('uploads', f'{voiceid}.mp3')
        audio_file.save(audio_file_path)
        audio = AudioSegment.from_mp3(audio_file_path)
        wav_file_path = os.path.join('uploads', f'{voiceid}.wav')
        # 音频频率统一转为48000hz
        audio = audio.set_frame_rate(48000)
        audio.export(wav_file_path, format='wav')
        audio_file_path = wav_file_path
    else:
        audio_file_path = os.path.join('uploads', f'{voiceid}.wav')
        audio_file.save(audio_file_path)

    executor.submit(handle_training, voiceid, audio_file_path)

    return jsonify({"code": 0, "msg": "OK", "data": {"voiceid": voiceid, "remainder": 180}}), 200

# 接口2: 获取训练状态
@app.route('/trainstate', methods=['POST'])
def train_state():
    voiceid = request.form.get('voiceid')

    with lock:
        if voiceid not in training_tasks:
            return jsonify({"code": 5, "msg": "操作失败", "data": {}}), 400

        task = training_tasks[voiceid]
    return jsonify({"code": task['status'], "msg": "OK", "data": {"voiceid": voiceid, "remainder": task['remainder']}}), 200

# 接口3: 模型推理
@app.route('/infer', methods=['POST'])
def perform_inference():
    voiceid = request.form.get('voiceid')
    voicetext = request.form.get('voicetext')
    print(f"收到请求，voiceid: {voiceid}")
    print(f"收到请求，voicetext: {voicetext}")

    if not voiceid or not voicetext:
        return jsonify({"code": 5, "msg": "缺少voiceid或voicetext", "data": {}}), 200

    @copy_current_request_context  # 关联上下文
    def handle_inference(voicetext,voiceid):
        try:
            # 模型推理
            audio_file_path = infer(voicetext, voiceid)
            if not audio_file_path:
                return jsonify({"code": 5, "msg": "模型推理失败", "data": {}})

            return jsonify({"code": 0, "msg": "OK", "data": {"voiceid": voiceid, "url": audio_file_path}})
        except Exception as e:
            logging.error("推理过程中出错: %s", e)
            return jsonify({"code": 5, "msg": "模型推理失败", "data": {}})

    # 收到的请求放入队列中，依次处理
    future = executor.submit(handle_inference, voicetext,voiceid)
    result = future.result()

    return result

if __name__ == '__main__':
    app.run(port=8001, host='0.0.0.0', use_reloader=False)

# 接口8001

# 声音克隆

# 环境：cosyvoice

# 添加显存回收处理模块 

# 音频要求3s-10s，读完有个停顿 克隆出的效果最好

# 删除MP3转wav模块，改为不同格式音频统一处理模块

# 添加队列中人数显示

# 添加训练剩余时间显示(训练完成后 genstate貌似有点慢，需要排查原因：去掉time.sleep(50))

# 貌似infer有问题（已解决）

# 解决infer推理队列问题

# 返回报错状态给jsonify

import os
import time
import queue
import logging
import threading
import datetime
from pydub import AudioSegment
from pydub.utils import mediainfo
from clone import clone_voice, infer  # 导入 train 和 infer 函数
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, copy_current_request_context
import uuid
app = Flask(__name__)


train_queue = queue.Queue()
train_lock = threading.Lock()

executor = ThreadPoolExecutor(max_workers=1)  # 确保同一时间只有一个训练任务在运行


# 存储训练任务的状态
training_tasks = {}  # key: voiceid, value: {'status': code, 'remainder': seconds}
# 存储任务开始处理的时间，用于动态计算剩余时间
task_start_times = {}

logging.basicConfig(level=logging.INFO)

def convert_to_standard_wav(input_path, output_path):
    # 检查音频信息
    info = mediainfo(input_path)

    # 定义格式所需的详细信息
    tar_codec = "pcm_s16le"
    tar_sample_fmt = "s16"
    target_sample_rate = 48000
    target_channels = 1  # 单声道，如果想要立体声，设置为2

    # 确定是否需要转换
    needs_conversion = (
        info['codec_name'] != tar_codec or
        info['sample_fmt'] != tar_sample_fmt or
        info['sample_rate'] != str(target_sample_rate) or
        info['channels'] != target_channels
    )

    if needs_conversion:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(target_sample_rate)
        audio = audio.set_sample_width(2)  # 16位是2字节
        audio = audio.set_channels(target_channels)
        audio.export(output_path, format="wav", codec=tar_codec)
        logging.info(f"音频已转换为标准格式: {output_path}")
    else:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        logging.info(f"音频已经是标准格式: {output_path}")
        
# 训练处理函数，更新状态并记录开始时间
def handle_training(voiceid, audio_file_path):
    try:
        # 从队列中取出当前任务
        train_queue.get()
        print(f"Task {voiceid} 队列取出，开始处理任务,当前北京时间为{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        # 更新任务状态为“处理中”，记录开始时间
        with train_lock:
            training_tasks[voiceid] = {'status': 1, 'remainder': 50, 'error_message': ''}
            task_start_times[voiceid] = time.time()

        # 执行训练过程
        train_status, _ = clone_voice(audio_file_path, voiceid)


        if train_status != '训练完成':
            with train_lock:
                training_tasks[voiceid] = {'status': 5, 'remainder': 0, 'error_message': train_status}
            logging.error(f"任务 Task {voiceid} 在训练过程中失败")
            return

        # 任务完成
        with train_lock:
            training_tasks[voiceid] = {'status': 0, 'remainder': 0, 'error_message': 'ok'}
            task_start_times.pop(voiceid, None)
        logging.info(f"任务 Task {voiceid} 成功完成训练")
    except Exception as e:
        logging.error(f"训练过程中出错: {e}")
        with train_lock:
            training_tasks[voiceid] = {'status': 5, 'remainder': 0,'error_message': str(e)}
            task_start_times.pop(voiceid, None)
            
  
# 更新训练任务提交逻辑：将任务加入队列和状态表
@app.route('/trainvoice', methods=['POST'])
def train_voice():
    voiceid = request.form.get('voiceid')
    audio_file = request.files.get('voicefile')

    if not voiceid :
        return jsonify({"code": 5, "msg": "却少音频ID", "data": {}}), 200

    if not audio_file:
        return jsonify({"code": 5, "msg": "缺少音频文件", "data": {}}), 200

    # 确保 uploads 文件夹存在
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # 保存音频文件
    input_path = os.path.join('uploads', f"{voiceid}_{audio_file.filename}")
    audio_file.save(input_path)

    # 转换为标准音频格式
    output_path = os.path.join('uploads', f"{voiceid}_standard.wav")
    convert_to_standard_wav(input_path, output_path)

    # 添加任务到队列并初始化状态
    with train_lock:
        training_tasks[voiceid] = {'status': 1, 'remainder': 50}  # 初始化任务状态
        train_queue.put(voiceid)  # 添加到队列
        print(f"Task:{voiceid} 添加到队列,当前北京时间为 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 提交异步任务
    executor.submit(handle_training, voiceid, output_path)

    return jsonify({"code": 0, "msg": "OK", "data": {"voiceid": voiceid, "remainder": 50}}), 200

# 接口2: 获取训练状态
@app.route('/trainstate', methods=['POST'])
def train_state():
    voiceid = request.form.get('voiceid')

    with train_lock:
        # 如果任务未在状态表中记录，返回失败状态
        if voiceid not in training_tasks:
            return jsonify({"code": 5, "msg": "操作失败", "data": {}}), 200

        # 获取任务状态
        task = training_tasks[voiceid]
        current_status = task['status']
        error_message = task['error_message']

        # 初始化 queueNumber 和 remainder
        queue_number = -1
        remainder = 0

        # 计算当前正在处理的任务剩余时间
        active_task_remainder = 0
        if task_start_times:
            for active_voiceid in task_start_times:
                elapsed_time = time.time() - task_start_times[active_voiceid]
                active_task_remainder = max(50 - int(elapsed_time), 0)
                break  # 只有一个任务在处理，退出循环

        if current_status == 1:  # 排队或处理中
            # 获取队列中的任务
            queue_list = list(train_queue.queue)
            if voiceid in queue_list:
                # 任务还在队列中
                queue_number = queue_list.index(voiceid) + 1
                remainder = queue_number * 50 + active_task_remainder  # 动态计算总剩余时间
            else:
                # 任务正在处理中
                queue_number = 0
                if voiceid in task_start_times:
                    elapsed_time = time.time() - task_start_times[voiceid]
                    remainder = max(50 - int(elapsed_time), 0)
                else:
                    remainder = 50  # 默认剩余时间

        elif current_status == 0:  # 已处理完成
            queue_number = -1
            remainder = 0

        elif current_status == 5:  # 失败
            queue_number = -1
            remainder = 0

        # 返回适配的 code
        code = 1 if current_status in [1] else current_status

    return jsonify({
        "code": code,
        "msg": error_message,
        "data": {
            "voiceid": voiceid,
            "remainder": remainder,
            "queueNumber": queue_number
        }
    }), 200
  



infer_queue = queue.Queue()
infer_lock = threading.Lock()

queue_pool = ThreadPoolExecutor(1)


# 存储推理任务状态
inference_results = {} 


def inference_task(voiceid, voicetext):
    
    print(f"开始处理推理任务 {voiceid}, 当前时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        result_path = infer(voicetext, voiceid)
        print(f"=========================================================================================")
        with infer_lock:
            if result_path:
                inference_results[voiceid] = {"voiceid": voiceid, "url": result_path}
            else:
                inference_results[voiceid] = {"voiceid": voiceid, "url": "notexist"}
                logging.error(f"推理任务 {voiceid} 失败")
    except Exception as e:
        logging.error(f"推理任务 {voiceid} 出现异常: {e}")
        with infer_lock:
            inference_results[voiceid] = {"voiceid": voiceid, "url": "notexist"}
            
def process_inference_task():
    """后台线程：从队列中取出推理任务，然后提交给线程池执行"""
    
    while True:
        voiceid, voicetext = infer_queue.get()
        try:
            print(f"拿取推理任务 {voiceid}, 当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            # 将推理任务提交到线程池
            queue_pool.submit(inference_task, voiceid, voicetext)
        except Exception as e:
            logging.error(f"推理任务 {voiceid} 提交到线程池失败: {e}")
        finally:
            infer_queue.task_done()
            
            
# 使用线程启动 process_inference_task 常驻响应推理请求
inference_thread = threading.Thread(target=process_inference_task, daemon=True)
inference_thread.start()


# queue_pool.submit(process_inference_task) # 使用线程启动 process_inference_task 常驻响应推理请求 需要设置queue_pool = ThreadPoolExecutor(2)



# 接口3: 模型推理
@app.route('/infer', methods=['POST'])
def perform_inference():
    voiceid = request.form.get('voiceid')
    voicetext = request.form.get('voicetext')
    print(f"收到请求，voiceid: {voiceid}")
    print(f"收到请求，voicetext: {voicetext}")

    if not voiceid or not voicetext:
        return jsonify({"code": 5, "msg": "缺少voiceid或voicetext", "data": {}}), 200

    # 检查任务是否已经存在结果
    if voiceid in inference_results:
        return jsonify(inference_results[voiceid]), 200

    with infer_lock:
        # 将任务添加到队列
        infer_queue.put((voiceid, voicetext))
        print(f"推理任务 {voiceid} 已添加到队列，当前北京时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，队列大小: {infer_queue.qsize()}")
    
    # 等待任务完成
    while voiceid not in inference_results:
        time.sleep(0.1)  # 小睡眠，减少 CPU 占用

    # 返回任务结果
    result = inference_results.pop(voiceid)  # 获取并移除结果
    
    return jsonify({
        "code": 0, 
        "msg": "OK", 
        "data": result
        }), 200


if __name__ == '__main__':
    app.run(threaded=True,port=8001, host='0.0.0.0', use_reloader=False)
       

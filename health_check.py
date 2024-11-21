# 程序健康保活机制

# 设置超时时间20秒

# 最大连续超时次数3次

import datetime
import requests
import os
import time

# 配置 Flask 服务的 URL 和测试参数
SERVICES = [
    {"url": "http://localhost:8001/trainstate", "params": {"voiceid": "test"}}
]

# 配置重启命令
SERVICE_RESTART_COMMANDS = {
    8001: {
        "name": "Voice Flask Service",
        "conda_env_path": "/home/wzhpc/anaconda3/envs/cosyvoice/bin/python",
        "script_path": "/opt/jyd01/wangruihua/api/digital/cosyvoice/voice_flask.py",
        "cuda_device": "1"
    }
}

# 配置日志目录
LOG_DIR = "/opt/jyd01/wangruihua/api/digital/cosyvoice/logs"

# 设置超时时间和最大重试次数
TIMEOUT = 20  # 单次请求超时时间（秒）
MAX_RETRIES = 3  # 最大连续超时次数


def check_service(url, params):
    """检查服务是否正常运行"""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.post(url, json=params, timeout=TIMEOUT)
            if response.status_code == 200:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"服务 {url} 正常——————{current_time}")
                return True
            else:
                print(f"服务 {url} 返回错误状态码: {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            retries += 1
            print(f"服务 {url} 请求超时，第 {retries} 次重试...")
        except Exception as e:
            print(f"服务 {url} 异常: {e}")
            return False
    print(f"服务 {url} 超时达到最大重试次数，判定为异常")
    return False


def kill_service_by_port(port):
    """通过端口号终止服务进程"""
    try:
        # 查找占用指定端口的进程
        find_command = f"lsof -i:{port} | awk '{{if(NR>1) print $2}}'"
        pid = os.popen(find_command).read().strip()
        if pid:
            os.system(f"kill -9 {pid}")
            print(f"成功终止运行的服务进程: {pid} (端口: {port})")
        else:
            print(f"端口 {port} 没有找到运行中的服务进程")
    except Exception as e:
        print(f"通过端口终止服务时发生异常: {e}")


def validate_paths(port):
    """验证关键路径是否存在"""
    if port not in SERVICE_RESTART_COMMANDS:
        raise KeyError(f"SERVICE_RESTART_COMMANDS 缺少端口 {port} 的配置")
    
    restart_info = SERVICE_RESTART_COMMANDS[port]
    conda_env_path = restart_info["conda_env_path"]
    script_path = restart_info["script_path"]

    if not os.path.exists(conda_env_path):
        raise FileNotFoundError(f"Conda 环境路径不存在：{conda_env_path}")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"脚本路径不存在：{script_path}")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"日志目录不存在，已创建：{LOG_DIR}")


def restart_service(port):
    """重启服务"""
    try:
        if port not in SERVICE_RESTART_COMMANDS:
            print(f"端口 {port} 缺少重启配置，跳过...")
            return
        
        restart_info = SERVICE_RESTART_COMMANDS[port]
        conda_env_path = restart_info["conda_env_path"]
        script_path = restart_info["script_path"]
        cuda_device = restart_info["cuda_device"]

        # 杀死旧进程
        kill_service_by_port(port)

        # 构造重启命令
        restart_command = (
            f"CUDA_VISIBLE_DEVICES={cuda_device} "
            f"{conda_env_path} {script_path} --port {port}"
        )
        log_path = os.path.abspath(os.path.join(LOG_DIR, f"nohup_{port}.log"))

        print(f"尝试重启服务: {restart_command}")
        os.system(f"nohup bash -c '{restart_command}' >> {log_path} 2>&1 &")
        print(f"服务 {port} 已重启，日志记录到: {log_path}")

        # 等待服务重新启动
        url = f"http://localhost:{port}/trainstate"
        if not wait_for_service(url):
            print(f"服务 {port} 重启失败，请检查日志")
    except Exception as e:
        print(f"重启服务 {port} 时出现异常: {e}")


def wait_for_service(url, max_wait=30):
    """等待服务重新启动并可用"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.post(url, timeout=5)
            if response.status_code == 200:
                print(f"服务 {url} 已重新启动并正常运行")
                return True
        except Exception:
            time.sleep(1)  # 等待1秒后重试
    print(f"服务 {url} 在重启后仍不可用")
    return False


# 检查每个服务并根据情况重启
for service in SERVICES:
    url = service["url"]
    params = service["params"]
    port = int(url.split(":")[2].split("/")[0])  # 从 URL 中提取端口号
    if not check_service(url, params):
        print(f"服务 {port} 异常，尝试重启...")
        validate_paths(port)
        restart_service(port)

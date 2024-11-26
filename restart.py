# 8001 接口重启

# 重启（声音克隆）服务restart.py


import os
import subprocess

# 配置服务的路径和环境
PORT = 8001
CONDA_ENV_PATH = "/home/wzhpc/anaconda3/envs/cosyvoice/bin/python"
SCRIPT_PATH = "/opt/jyd01/wangruihua/api/digital/cosyvoice/voice_flask.py"
CUDA_DEVICE = "6"
LOG_DIR = "/opt/jyd01/wangruihua/api/digital/cosyvoice/logs"
LOG_PATH = os.path.join(LOG_DIR, f"nohup_{PORT}.log")


def kill_port(port):
    """杀掉占用指定端口的进程"""
    try:
        # 获取占用端口的进程 ID
        result = subprocess.run(
            f"lsof -ti:{port}", shell=True, capture_output=True, text=True
        )
        pid = result.stdout.strip()
        if pid:
            os.system(f"kill -9 {pid}")
            print(f"已杀掉占用端口 {port} 的进程，PID: {pid}")
        else:
            print(f"端口 {port} 没有被占用")
    except Exception as e:
        print(f"杀掉端口 {port} 的进程时发生异常: {e}")


def restart_service():
    """重启 8001 服务"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 构造启动命令
        restart_command = (
            f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE} "
            f"{CONDA_ENV_PATH} {SCRIPT_PATH} --port {PORT}"
        )
        
        # 使用 nohup 启动服务
        os.system(f"nohup bash -c '{restart_command}' >> {LOG_PATH} 2>&1 &")
        print(f"服务 {PORT} 已重启，日志记录到: {LOG_PATH}")
    except Exception as e:
        print(f"重启服务 {PORT} 时发生异常: {e}")


if __name__ == "__main__":
    print(f"开始检查并重启端口 {PORT} 的服务...")
    kill_port(PORT)  # 杀掉占用的端口
    restart_service()  # 重启服务

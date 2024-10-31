# 提取MP3音频的前9s

import os
import time
from pydub import AudioSegment
from pydub.silence import split_on_silence

# 定义输入和输出文件路径
input_file = "/opt/jyd01/wangruihua/api/digital/cosyvoice/参考音频/曹总原音.mp3"
output_file = "/opt/jyd01/wangruihua/api/digital/cosyvoice/参考音频/曹总原音_前9s.mp3"

# 加载音频文件
audio = AudioSegment.from_mp3(input_file)

# 提取前9秒的音频
audio = audio[:9000]

# 保存提取的音频文件
audio.export(output_file, format="mp3")
print(f"提取的音频已保存到 {output_file}")

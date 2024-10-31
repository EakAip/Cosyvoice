# 调用7998端口进行音频转写 统计语速

from gradio_client import Client, file
from pydub import AudioSegment
import re


audio_path = "/opt/jyd01/wangruihua/api/digital/cosyvoice/参考音频/王双原音.wav"

client = Client("http://188.18.18.106:7998/")
result = client.predict(
		audio_file=file(audio_path),
		hotwords="Hello!!",
		api_name="/recognize_audio"
)
# 使用正则，result 中将只保留 speaker0: 之后的内容。
result = re.sub(r'.*speaker0:', '', result[0])
print(result)

print(f"字数：{len(result)}")


audio = AudioSegment.from_wav(audio_path)

time = len(audio) / 1000
print(f"时长：{time}")


print(f"语速:{len(result)/time}")
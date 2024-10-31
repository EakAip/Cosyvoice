# mp3转wav

from pydub import AudioSegment

# Load the original MP3 file
input_path = "/mnt/data/曹总原音.mp3"
output_path = "/mnt/data/曹总原音_调整.wav"

# Convert the MP3 to WAV with matching parameters: 48000 Hz sample rate, 2 channels
audio = AudioSegment.from_mp3(input_path)
audio_resampled = audio.set_frame_rate(48000).set_channels(2)

# Export the modified audio
audio_resampled.export(output_path, format="wav")

output_path

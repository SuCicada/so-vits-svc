from pydub import AudioSegment

# 读取第一个声音文件
audio1 = AudioSegment.from_file("out.wav")

# 读取第二个声音文件
audio2 = AudioSegment.from_file("1_Bôa - Duvet TV Sized_(Instrumental).wav")


# 检查两个音频文件的长度，选择最长的音频作为基准
max_length = max(len(audio1), len(audio2))

# 将两个音频文件进行同步
audio1 = audio1[:max_length]
audio2 = audio2[:max_length]

# 将两个音频文件进行合并
combined = audio1.overlay(audio2)

# 保存合并后的音频文件
combined.export("merged_audio.wav", format="wav")

import simpleaudio as sa

# 加载WAV文件
wave_obj = sa.WaveObject.from_wave_file("merged_audio.wav")

# 播放WAV文件
play_obj = wave_obj.play()

# 等待播放完成
play_obj.wait_done()

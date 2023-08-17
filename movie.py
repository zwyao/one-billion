import os

import speech_recognition as sr
import moviepy.editor as mp
import whisper

# from pathlib import Path
# import os
#
# from pydub import AudioSegment
# import wave
# import contextlib
# import datetime
# import pydub
# from pydub.silence import split_on_silence
# from tqdm.notebook import tqdm

# def get_large_audio_transcription(recognizer,path):
# """
#     Splitting the large audio file into chunks
#     and apply speech recognition on each of these chunks
#     """
# # open the audio file using pydub
#     sound = AudioSegment.from_wav(path)
# # split audio sound where silence is 700 miliseconds or more and get chunks
#     chunks = split_on_silence(sound,
# # experiment with this value for your target audio file
#         min_silence_len = 500,
# # adjust this per requirement
#         silence_thresh = sound.dBFS-14,
# # keep the silence for 1 second, adjustable as well
#         keep_silence=500,
#     )
#     folder_name = "audio-chunks"
# # create a directory to store the audio chunks
# if not os.path.isdir(folder_name):
#         os.mkdir(folder_name)
#     whole_text = []
#     time_lines = []
# # process each chunk
#     start_time = datetime.datetime.fromisoformat('2022-01-01T00:00:00')
#
# for i, audio_chunk in enumerate(chunks, start=1):
# # export audio chunk and save it in
# # the `folder_name` directory.
#         chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
#         audio_chunk.export(chunk_filename, format="wav")
#
# # recognize the chunk
# with sr.AudioFile(chunk_filename) as source:
#             audio_listened = recognizer.record(source)
# # try converting it to text
# try:
#                 text = recognizer.recognize_google(audio_listened,language="en-us")
# except sr.UnknownValueError as e:
# #print("Error:", str(e))
# pass
# else:
#                 text = f"{text.capitalize()}. "
# #print(start_time.time(), ":", text)
# #whole_text += text
#                 whole_text.append(text)
#                 time_lines.append(start_time)
#         duration = get_audio_duration(chunk_filename)
#         start_time += datetime.timedelta(seconds=duration)
# # return the text for all chunks detected
# return whole_text,time_lines

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 视频到音频转换
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    clip = mp.VideoFileClip(r"视频转文字.mp4")
    clip.audio.write_audiofile(r"converted.wav")

    model = whisper.load_model("large")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("converted.wav")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16=False, language="zh")
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)
    exit(os.EX_OK)

    # 定义识别器
    r = sr.Recognizer()

    # 导入在上一步（步骤2）中创建的音频文件
    audio = sr.AudioFile("converted.wav")
    with audio as source:
        audio_file = r.record(source)
    result = r.recognize_google(audio_file, language="zh-CN")

    # exporting the result
    with open('recognized.txt', mode='w') as file:
        file.write("Recognized Speech:")
        file.write("\n")
        file.write(result)
        print("ready!")

import speech_recognition as sr
import noisereduce as nr
import numpy as np
import wave
import os

def capture_audio(filename="temp.wav"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio_data = r.listen(source)
        print("Audio captured.")
    
    with open(filename, "wb") as f:
        f.write(audio_data.get_wav_data())
    
    return filename

def read_wave(filename):
    with wave.open(filename, 'rb') as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio_data_np = np.frombuffer(frames, dtype=np.int16)
    return audio_data_np, sample_rate

def reduce_noise(audio_data_np, sample_rate):
    noise_reduced = nr.reduce_noise(y=audio_data_np, sr=sample_rate)
    return noise_reduced

def recognize_speech(audio_data_np, sample_rate):
    # Convert numpy array back to audio data
    temp_file = "temp_reduced.wav"
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data_np.tobytes())
    
    r = sr.Recognizer()
    with sr.AudioFile(temp_file) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Could not understand audio"
        except sr.RequestError:
            text = "Could not request results from Google Speech Recognition service"
    
    os.remove(temp_file)
    return text

if __name__ == "__main__":
    # Capture audio and save to temp.wav
    audio_filename = capture_audio()
    
    # Read the captured audio
    audio_data_np, sample_rate = read_wave(audio_filename)
    
    # Reduce noise in the audio data
    noise_reduced_np = reduce_noise(audio_data_np, sample_rate)
    
    # Recognize speech from the noise-reduced audio data
    recognized_text = recognize_speech(noise_reduced_np, sample_rate)
    print("Recognized Text:", recognized_text)
    
    # Clean up temporary audio file
    os.remove(audio_filename)

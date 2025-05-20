# Usage:
# python dnsmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.csv -p

import argparse
import concurrent.futures # Used to parallelise tasks
import glob # Used to find all file paths that match a specified pattern
import os

import librosa # Used for audio and music analysis
import numpy as np
import numpy.polynomial.polynomial as poly 
import onnxruntime as ort # Used for running ML models in Open Neural Network Exchange (ONNX) format
import pandas as pd
import soundfile as sf # Used for reading and writing sound files (e.g. in .wav format)
from requests import session 
from tqdm import tqdm
import random

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
    
    # Generates a Mel spectogram (frequency against time graph) for an input audio signal
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40 # Converts the power spectogram (linear scale) to dB and normalise
        return mel_spec.T # Transposed so that each row corresponds to a time step and each column corresponds to a Mel frequency bin

    # Computes signal quality, background noise quality and overall quality scores using predefined polynomials
    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    # Evaluates audio quality and returns the scores
    def __call__(self, fpath, sampling_rate, is_personalized_MOS):
        aud, input_fs = sf.read(fpath) # aud = audio data as a np array; input_fs = sampling rate of audio file
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs) # Resample to the desired sampling rate
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples: # Adjust audio length
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0] # Generate ground truth human ratings using ITU-T P.808
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0] # Generate raw MOS scores using the primary model
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,is_personalized_MOS) # Adjust MOS scores by fitting to predefined polynomials
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs}
        # clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs, 'sr':fs}
        # clip_dict['num_hops'] = num_hops
        # Raw scores
        # clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        # clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        # clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        # Adjusted scores
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        # Ground truth
        clip_dict['P808_MOS'] = np.mean(predicted_p808_mos)
        return clip_dict

def main(args):
    datasets = glob.glob(os.path.join(args.testset_dir, "*"))
    audio_clips_list = []
    p808_model_path = os.path.join('DNSMOS', 'model_v8.onnx')

    if args.personalized_MOS:
        primary_model_path = os.path.join('pDNSMOS', 'sig_bak_ovr.onnx')
    else:
        primary_model_path = os.path.join('DNSMOS', 'sig_bak_ovr.onnx')

    compute_score = ComputeScore(primary_model_path, p808_model_path)

    rows = []
    clips = []
    clips = glob.glob(os.path.join(args.testset_dir, "*.wav"))
    is_personalized_eval = args.personalized_MOS
    desired_fs = SAMPLING_RATE
    for d in tqdm(datasets):
        max_recursion_depth = 10
        audio_path = d
        audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
        while len(audio_clips_list) == 0 and max_recursion_depth > 0:
            audio_path = os.path.join(audio_path, "**")
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
            max_recursion_depth -= 1
        clips.extend(audio_clips_list)

    # Randomly select 10 audio samples
    if len(clips) > 10:
        clips = random.sample(clips, 10)

    # Parallelise score computation across the audio clips
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # executor.submit submits tasks to the thread pool
        # Each task calls the compute_score function with the arguments clip, desired_fs and is_personalised_eval
        future_to_url = {executor.submit(compute_score, clip, desired_fs, is_personalized_eval): clip for clip in clips}
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (clip, exc))
            else:
                rows.append(data)            

    df = pd.DataFrame(rows)
    if args.csv_path:
        csv_path = args.csv_path
        df.to_csv(csv_path)
    else:
        print(df.describe())

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--testset_dir", default='.', 
                        help='Path to the dir containing audio clips in .wav to be evaluated')
    parser.add_argument('-o', "--csv_path", default=None, help='Dir to the csv that saves the results')
    parser.add_argument('-p', "--personalized_MOS", action='store_true', 
                        help='Flag to indicate if personalized MOS score is needed or regular')
    
    args = parser.parse_args()

    main(args)

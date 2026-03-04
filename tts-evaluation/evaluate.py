"""
uv venv --python 3.12
source .venv/bin/activate

uv pip install vllm datasets transformers torch soundfile jiwer
uv pip install neucodec
uv pip install git+https://github.com/sarulab-speech/UTMOSv2.git
"""
from vllm import LLM
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor
)
import argparse
import os
import logging
import torch
from neucodec import NeuCodec
import re
import soundfile as sf
import requests 
from tqdm import tqdm
import glob
import torchaudio.transforms as T
from jiwer import cer
import utmosv2 
from typing import Union
from time import time
from torch.nn.utils.rnn import pad_sequence

# def run_whisper(model_name:str):
#     # Start the vLLM server using system
#     os.system(f"vllm serve --model {model_name}")

def prepare_input_tokens(
    speaker_name: str, 
    target_text: Union[str, list[str]], 
    description: Union[str, list[str]], 
    tokenizer,
)->dict: 
    if isinstance(target_text, str):
        input_text = f"<|im_start|>{speaker_name}: {target_text}<|description|>{description}<|speech_start|>"
        return tokenizer(input_text, return_tensors="pt")
    else: 
        input_texts = []
        for text, desc in zip(target_text, description):
            input_texts.append(f"<|im_start|>{speaker_name}: {text}<|description|>{desc}<|speech_start|>")
        return tokenizer(input_texts, return_tensors="pt", padding=True)

def decocde_output_tokens(
    output_tokens: torch.Tensor, # (B, S)
    tokenizer,
    codec,
    save_path: Union[str, list[str]] = None,
): 
    "Decode output tokens to audio and save the audio file"
    if output_tokens.shape[0] > 1 and not isinstance(save_path, list) and len(save_path) != output_tokens.shape[0]:
        raise ValueError("save_path should be a list of paths with the same length as output_tokens batch size")
    
    decode_tokens = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
    for i, decode_token in enumerate(decode_tokens):
        codec_tokens = re.findall(r"<\|s_(\d+)\|>", decode_token)
        codec_tokens = torch.tensor([int(token) for token in codec_tokens])[None, None, :].to(codec.device)
        with torch.no_grad():
            audio = codec.decode_code(codec_tokens).to('cpu') # (1, 1, T)
            sf.write(save_path[i] if isinstance(save_path, list) else save_path, audio[0,0].numpy(), samplerate=24000)
            # sf.write(save_path, audio[0,0].numpy(), samplerate=24000)
    
def main(
    dataset: str,
    model_name: str,
    output_dir: str, 
    length: int = None, 
    batch_size: int = 1,
):
    ds_dict = load_dataset(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    codec = None
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate TTS 
    timer = time()
    for lang, ds in ds_dict.items():
        ds = ds.remove_columns(["reference_audio"])
        ds = ds.select(range(length)) if length is not None else ds
        
        for start in tqdm(range(0, len(ds), batch_size), desc=f"Generating TTS for {lang}", total=(len(ds) + batch_size - 1) // batch_size):
            end = min(start + batch_size, len(ds))
            samples = ds[start:end]
            save_paths = [os.path.join(output_dir, f"output_{lang}_{id}.wav") for id in samples["id"]]
            if all(os.path.exists(path) for path in save_paths):
                continue
            
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
                codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
                codec.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
                
            input_tokens = prepare_input_tokens(
                speaker_name="genshin-voice_audio_Rahman", 
                target_text=samples["text"], 
                description=samples["APS"],
                tokenizer=tokenizer
            ).to(device)
            
            with torch.no_grad():
                output_tokens = model.generate(
                    **input_tokens, 
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.8,
                    repetition_penalty=1.15,)

            decocde_output_tokens(
                output_tokens=output_tokens, 
                tokenizer=tokenizer,
                codec=codec, 
                save_path=save_paths
            )
        
    # clear GPU memory and codec model
    model = None
    del codec
    torch.cuda.empty_cache()
    print(f"TTS generation completed in {time() - timer:.2f} seconds.")
    timer = time()
    
    # 2. Evaluate TTS using ASR (Whisper)   
    for lang, ds in ds_dict.items():
        ds = ds.remove_columns(["reference_audio"])
        ds = ds.select(range(length)) if length is not None else ds
        
        for start in tqdm(range(0, len(ds), batch_size), desc=f"Transcription for {lang}", total=(len(ds) + batch_size - 1) // batch_size):
            end = min(start + batch_size, len(ds))
            samples = ds[start:end]
            save_paths = [os.path.join(output_dir, f"trans_{lang}_{id}.txt") for id in samples["id"]]
            audio_paths = [os.path.join(output_dir, f"output_{lang}_{id}.wav") for id in samples["id"]]
            if all(os.path.exists(save_path) for save_path in save_paths):
                continue
    #         save_path = os.path.join(output_dir, f"trans_{lang}_{sample['id']}.txt")
    #         audio_path = os.path.join(output_dir, f"output_{lang}_{sample['id']}.wav")
    #         if not os.path.exists(audio_path):
    #             continue
    #         if os.path.exists(save_path):
    #             continue
            if model is None:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    "openai/whisper-large-v3", 
                    dtype=torch.bfloat16, 
                ).to(device)
                processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
                
            # load audio and preprocess
            audios = []
            for audio_path in audio_paths:
                audio, sr = sf.read(audio_path)
                audio = torch.from_numpy(audio).to(torch.float32)
                print(audio.shape)
                audios.append(audio)
            audios = pad_sequence(audios, batch_first=True) # (B, T)
            
            sampler = T.Resample(orig_freq=24000, new_freq=16000)
            audios = sampler(audios) # (B, T')
            
            inputs = processor(
                audios.numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad(), torch.autocast(device_type=device.type):
                predicted_ids = model.generate(**inputs)
            
            transcription = processor.batch_decode(predicted_ids.cpu().numpy(), skip_special_tokens=True)
            
            for text, save_path in zip(transcription, save_paths):
                with open(save_path, "w") as f:
                    f.write(text.strip())
    
    model = None 
    torch.cuda.empty_cache()
    print(f"ASR evaluation completed in {time() - timer:.2f} seconds.")
    timer = time()
    
    # # print("Run Whisper ASR evaluation...")
    
    # # 3. Evaluate MOS using UTMOSv2
    for lang, ds in ds_dict.items():
        ds = ds.remove_columns(["reference_audio"])
        ds = ds.select(range(length)) if length is not None else ds
        
        for start in tqdm(range(0, len(ds), batch_size), desc=f"Transcription for {lang}", total=(len(ds) + batch_size - 1) // batch_size):
            end = min(start + batch_size, len(ds))
            samples = ds[start:end]
            save_paths = [ os.path.join(output_dir, f"mos_{lang}_{id}.txt") for id in samples["id"]]
            audio_paths = [ f"output_{lang}_{id}.wav" for id in samples["id"] ]
    #         save_path = os.path.join(output_dir, f"mos_{lang}_{sample['id']}.txt")
            if all(os.path.exists(save_path) for save_path in save_paths):
                continue
            
            if model is None:
                model = utmosv2.create_model(pretrained=True, device=device)
            mos = model.predict(input_dir=output_dir, val_list=audio_paths )
            for prediction, save_path in zip(mos, save_paths):
                with open(save_path, "w") as f:
                    f.write(str(prediction["predicted_mos"]))
    print(f"MOS evaluation completed in {time() - timer:.2f} seconds.")
    model = None 
    
    # # 4. Summarize the evaluation results
    cer_scores = []
    mos_scores = []
    for lang, ds in ds_dict.items():
        ds = ds.remove_columns(["reference_audio"])
        ds = ds.select(range(length)) if length is not None else ds
        
        for sample in tqdm(ds, desc=f"Calculating CER for {lang}", total=len(ds)):
            trans_path = os.path.join(output_dir, f"trans_{lang}_{sample['id']}.txt")
            mos_path = os.path.join(output_dir, f"mos_{lang}_{sample['id']}.txt")
            with open(trans_path, "r") as f:
                transcription = f.read().strip()
            with open(mos_path, "r") as f:
                mos = float(f.read().strip())
                
            cer_score = min(cer(sample["text"], transcription), 1.0)
            cer_scores.append(cer_score)
            mos_scores.append(mos)
            
    avg_cer = sum(cer_scores) / len(cer_scores)
    avg_mos = sum(mos_scores) / len(mos_scores)
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average MOS: {avg_mos:.4f}")
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Scicom-intl/Multilingual-Expressive-TTS-1.7B")
    parser.add_argument("--dataset", type=str, default="CaasiHUANG/InstructTTSEval")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    main(args.dataset, 
         args.model_name, 
         args.output_dir, 
         args.length, 
         args.batch_size)

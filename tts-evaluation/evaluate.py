"""
uv venv --python 3.12
source .venv/bin/activate

uv pip install datasets transformers torch soundfile jiwer  # setuptools git+https://github.com/resemble-ai/chatterbox.git git+https://github.com/resemble-ai/Perth.git
uv pip install neucodec
uv pip install git+https://github.com/sarulab-speech/UTMOSv2.git
uv pip install git+https://github.com/resemble-ai/chatterbox.git (for multilingual TTS)

# evaluate TTS model with pass@k, k=3
python evaluate.py \
    --model_name Scicom-intl/Multilingual-Expressive-TTS-1.7B \
    --batch_size 2 \
    --sampling \
    --sample_size 2 \
    --output_dir ./scicom-tts \
    --length 4
    
python evaluate.py \
    --model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --batch_size 2 \
    --sampling \
    --sample_size 2 \
    --length 10
    
python evaluate.py \
    --model_name chatterbox \
    --batch_size 2 \
    --length 10 \
    --sampling \
    --sample_size 2
    
hf download Scicom-intl/Evaluation-Multilingual-VC --exclude *.zip --repo-type dataset
"""
from datasets import load_dataset, DatasetDict, Dataset as HFDataset
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
import json

class BaseTTSModel: 
    def __init__(self,
                 model_name: str, 
                 device: torch.device, 
                 sampling: bool = False, 
                 sample_size: int = 3, ):
        pass 
    
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")
class ScicomTTSModel(BaseTTSModel): 
    def __init__(self, 
                 model_name: str, 
                 device: torch.device, 
                 sampling: bool = False, 
                 sample_size: int = 3, ):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
        self.sampling = sampling
        self.sample_size = sample_size
        self.device = device 
    
    @staticmethod
    def _prepare_input_tokens(
        target_text: Union[str, list[str]], 
        description: Union[str, list[str], None], 
        tokenizer,
        speaker_name: str = "genshin-voice_audio_Rahman"
    )->dict: 
        "Prepare input tokens for TTS generation."
        if isinstance(target_text, str):
            description_format = f"<|description|>{description}" if description is not None else ""
            input_text = f"<|im_start|>{speaker_name}: {target_text}{description_format}<|speech_start|>"
            return tokenizer(input_text, return_tensors="pt")
        else: 
            input_texts = []
            description = description if description is not None else [None] * len(target_text)
            for text, desc in zip(target_text, description):
                description_format = f"<|description|>{desc}" if desc is not None else ""
                input_texts.append(f"<|im_start|>{speaker_name}: {text}{description_format}<|speech_start|>")
            return tokenizer(input_texts, return_tensors="pt", padding=True)

    @staticmethod
    def _decocde_output_tokens(
        output_tokens: torch.Tensor, # (B, S)
        tokenizer,
        codec,
        save_path: Union[str, list[str]] = None,
    ): 
        "Decode output tokens to audio and save the audio file."
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
                
    def generate(self,  
                 target_text: Union[str, list[str]],
                 description: Union[str, list[str], None],
                 save_paths: list[str], 
                 **kwargs):
        
        speaker_name = kwargs.get("speaker_name", "genshin-voice_audio_Rahman")
        input_tokens = self._prepare_input_tokens(
            target_text=target_text, 
            description=description,
            tokenizer=self.tokenizer, 
            speaker_name=speaker_name,
        ).to(self.device)
        
        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens":2048,
                "do_sample":True,
                "temperature":0.8,
                "repetition_penalty":1.15,
            }
            if self.sampling: 
                generation_kwargs["num_return_sequences"] = self.sample_size
                generation_kwargs["num_beams"] = 1
            output_tokens = self.model.generate(
                **input_tokens, 
                **generation_kwargs
                )

        self._decocde_output_tokens(
            output_tokens=output_tokens, 
            tokenizer=self.tokenizer,
            codec=self.codec, 
            save_path=save_paths
        )
class QwenTTSModel(BaseTTSModel):
    def __init__(self,
                 model_name: str, 
                 device: torch.device, 
                 sampling: bool = False, 
                 sample_size: int = 3, ): 
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError("QwenTTSModel requires the qwen-tts package. Please install it first.")
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device.type,
            dtype=torch.bfloat16,
        )
        self.sampling = sampling
        self.sample_size = sample_size
    
    def mapped_language(self, lang):
        mapping = {
            "zh": "Chinese",
            "en": "English",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "ja": "Japanese",
            "ko": "Korean",
            "pt": "Portuguese",
            "ru": "Russian",
            "es": "Spanish"
        }
        return mapping.get(lang, lang)
    
    def supported_languages(self):
        return [
            'zh', 'en', 'fr', 'de', 'it', 'ja', 'ko', 'pt', 'ru', 'es'
        ]

    def generate(self,  
                 target_text: Union[str, list[str]],
                 description: Union[str, list[str], None],
                 save_paths: list[str], 
                 speaker_name: str = "Vivian",
                 **kwargs):
        
        lang = kwargs.get("language", "en")
        if lang not in self.supported_languages():
            return
        
        _target_text = []
        _description = []
        _speaker_name = [ speaker_name ]
        if self.sampling: 
            for text in target_text:
                _target_text.extend([text] * self.sample_size)
            for desc in description:
                _description.extend([desc] * self.sample_size)
            _speaker_name = _speaker_name * len(target_text) * self.sample_size
            _language = [self.mapped_language(lang)] * len(target_text) * self.sample_size
        else: 
            _target_text = target_text
            _speaker_name = _speaker_name * len(target_text)
            _description = description
            _language = [self.mapped_language(lang)] * len(target_text)

        wavs, sr = self.model.generate_custom_voice(
            text=_target_text,
            language=_language,
            speaker=_speaker_name,
            instruct=_description
        )
        if sr != 24000:
            raise ValueError(f"Expected sample rate of 24000, but got {sr}. Resampling is needed")
        
        for wav, save_path in zip(wavs, save_paths):
            sf.write(save_path, wav, samplerate=sr)
class ChatterBox(BaseTTSModel):
    def __init__(self, 
                 model_name: str, 
                 device: torch.device, 
                 sampling: bool = False, 
                 sample_size: int = 3, ):
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError:
            raise ImportError("ChatterBox model requires the chatterbox-tts package. Please install it first.")
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=device.type)
        self.sampling = sampling
        self.sample_size = sample_size
    
    def supported_languages(self):
        # Arabic (ar) • Danish (da) • German (de) • Greek (el) • 
        # English (en) • Spanish (es) • Finnish (fi) • French (fr) • 
        # Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • 
        # Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • 
        # Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)
        supported_languages = [
            "ar", "da", "de", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko",
            "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"
        ]
        return supported_languages

    def generate(self,  
                 target_text: Union[str, list[str]],
                 description: Union[str, list[str]],
                 save_paths: list[str], 
                 **kwargs):
        """
        Eg.
        multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
        wav = multilingual_model.generate(text, language_id="fr")
        ta.save("test-2.wav", wav, multilingual_model.sr)
        """
        if isinstance(target_text, list) and len(target_text) > 1:
            print("ChatterBox model currently only supports single inference. Falling back to single inference mode.")
            
        lang = kwargs.get("language", "en")
        if lang not in self.supported_languages():
            return 
        
        for i, text in enumerate(target_text):
            if self.sampling:
                for j in range(self.sample_size):
                    wav = self.model.generate(text, language_id=lang)
                    if self.model.sr != 24000:
                        raise ValueError(f"Expected sample rate of 24000, but got {self.model_sr}. Resampling is needed")
                    sf.write(save_paths[i * self.sample_size + j], wav[0].cpu().numpy(), samplerate=self.model.sr)
            else:
                wav = self.model.generate(text, language_id=lang)
                if self.model.sr != 24000:
                    raise ValueError(f"Expected sample rate of 24000, but got {self.model_sr}. Resampling is needed")
                sf.write(save_paths[i], wav[0].cpu().numpy(), samplerate=self.model.sr)

MODEL_MAPPING = {
    "Scicom-intl/Multilingual-Expressive-TTS-1.7B" : ScicomTTSModel,
    "Scicom-intl/Multilingual-Expressive-TTS-0.6B" : ScicomTTSModel,
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": QwenTTSModel,
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": QwenTTSModel, 
    "chatterbox": ChatterBox,
}

class Dataset:
    def __init__(self, 
                 dataset_name: str, 
                 length: int = None):
        self.ds = load_dataset(
            dataset_name, 
            "combine_filtered_whisper_large_v3", 
            split="train"
        )
        self.__split_by_language()
        self.__drop_unused_columns(["source_text", "upvotes", "speaker_id", "audio_filename"])
        self.ds = self.ds.map(lambda x, idx: {"id": idx}, with_indices=True)
        self.ds = self.ds.rename_column("target_text", "text")
        self.length = length
        self.expressive = False 
    
    def __get_language_mapping(self)->dict[str, str]:
        "Get the language code mapping to map with whisper supported languages."
        with open("common-voice-whisper-mapping.json", "r") as f:
            mapping = json.load(f)
        return mapping
    
    def __split_by_language(self, lang_columns: str = "language"):
        df = self.ds.to_pandas()
        # map language to whisper supported language, else filter out the unsupported language
        mapping = self.__get_language_mapping()
        df[lang_columns] = df[lang_columns].map(mapping)
        # print out the number of dropped samples due to unsupported language
        num_dropped = df[lang_columns].isna().sum()
        if num_dropped > 0:
            print(f"Dropping {num_dropped} samples due to unsupported language.")
        else:
            print("All samples have supported languages.")
        print(f"Remaining languages after mapping: {df[lang_columns].unique()}")
        df = df.dropna(subset=[lang_columns])
        groups = df.groupby(lang_columns)
        self.ds = DatasetDict(
            { lang: HFDataset.from_pandas(group.reset_index(drop=True)) for lang, group in groups }
        )
        
    def __drop_unused_columns(self, columns_to_drop: list[str]): 
        self.ds = self.ds.remove_columns(columns_to_drop)
        
    
    def __iter__(self):
        for lang, ds in self.ds.items():
            if self.length is not None: 
                ds = ds.select(range(self.length))
            yield lang, ds
    

def main(
    dataset: str,
    model_name: str,
    output_dir: str, 
    length: int = None, 
    batch_size: int = 1,
    sampling: bool = False, 
    sample_size: int = 3
):
    # ds_dict = load_dataset(dataset)
    dataset = Dataset(dataset, length=length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    codec = None
    os.makedirs(output_dir, exist_ok=True)
    
    if not model_name in MODEL_MAPPING:
        raise ValueError(f"Model {model_name} is not supported. Please choose from {list(MODEL_MAPPING.keys())}")
    
    # 1. Generate TTS 
    timer = time()
    for lang, ds in dataset:
        for start in tqdm(range(0, len(ds), batch_size), desc=f"Generating TTS for {lang}", total=(len(ds) + batch_size - 1) // batch_size):
            end = min(start + batch_size, len(ds))
            samples = ds[start:end]
            if sampling: 
                save_paths = []
                for id in samples["id"]:
                    save_paths.extend([os.path.join(output_dir, f"output_{lang}_{id}_{i}.wav") for i in range(sample_size)])
            else:
                save_paths = [os.path.join(output_dir, f"output_{lang}_{id}.wav") for id in samples["id"]]
            if all(os.path.exists(path) for path in save_paths):
                continue
            
            if model is None:
                model = MODEL_MAPPING[model_name](model_name=model_name, device=device, sampling=sampling, sample_size=sample_size)
                
            model.generate(
                target_text=samples["text"],
                description=samples.get("APS", None),
                save_paths=save_paths, 
                language=lang
            )
        
    # clear GPU memory and codec model
    model = None
    del codec
    torch.cuda.empty_cache()
    print(f"TTS generation completed in {time() - timer:.2f} seconds.")
    timer = time()
    
    # 2. Evaluate TTS using ASR (Whisper)
    for lang, ds in dataset:
        for start in tqdm(range(0, len(ds), batch_size), desc=f"Transcription for {lang}", total=(len(ds) + batch_size - 1) // batch_size):
            end = min(start + batch_size, len(ds))
            samples = ds[start:end]
            if sampling:
                save_paths = []
                audio_paths = []
                for id in samples["id"]:
                    save_paths.extend([os.path.join(output_dir, f"trans_{lang}_{id}_{i}.txt") for i in range(sample_size)])
                    audio_paths.extend([os.path.join(output_dir, f"output_{lang}_{id}_{i}.wav") for i in range(sample_size)])
            else:
                save_paths = [os.path.join(output_dir, f"trans_{lang}_{id}.txt") for id in samples["id"]]
                audio_paths = [os.path.join(output_dir, f"output_{lang}_{id}.wav") for id in samples["id"]]
            if all(os.path.exists(save_path) for save_path in save_paths):
                continue

            if model is None:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    "openai/whisper-large-v3", 
                    torch_dtype=torch.bfloat16, # for backward compatibility
                ).to(device)
                processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
                
            # load audio and preprocess
            audios = []
            for audio_path in audio_paths:
                audio, sr = sf.read(audio_path)
                audio = torch.from_numpy(audio).to(torch.float32)
                audios.append(audio)
            audios = pad_sequence(audios, batch_first=True) # (B, T)
            
            sampler = T.Resample(orig_freq=24000, new_freq=16000)
            audios = sampler(audios) # (B, T')
            
            inputs = processor(
                audios.numpy(), 
                sampling_rate=16000, 
                return_tensors="pt", 
                language=lang,
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
    for lang, ds in dataset:
        for start in tqdm(range(0, len(ds), batch_size), desc=f"Transcription for {lang}", total=(len(ds) + batch_size - 1) // batch_size):
            end = min(start + batch_size, len(ds))
            samples = ds[start:end]
            if sampling:
                save_paths = []
                audio_paths = []
                for id in samples["id"]:
                    save_paths.extend([os.path.join(output_dir, f"mos_{lang}_{id}_{i}.txt") for i in range(sample_size)])
                    audio_paths.extend([ f"output_{lang}_{id}_{i}.wav" for i in range(sample_size)])
            else:
                save_paths = [ os.path.join(output_dir, f"mos_{lang}_{id}.txt") for id in samples["id"]]
                audio_paths = [ f"output_{lang}_{id}.wav" for id in samples["id"] ]

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
    for lang, ds in dataset:
        for sample in tqdm(ds, desc=f"Calculating CER for {lang}", total=len(ds)):
            if sampling:
                for i in range(sample_size):
                    trans_path = os.path.join(output_dir, f"trans_{lang}_{sample['id']}_{i}.txt")
                    mos_path = os.path.join(output_dir, f"mos_{lang}_{sample['id']}_{i}.txt")
                    with open(trans_path, "r") as f:
                        transcription = f.read().strip()
                    with open(mos_path, "r") as f:
                        mos = float(f.read().strip())
                    cer_score = min(cer(sample["text"], transcription), 1.0)
                    cer_scores.append(cer_score)
                    mos_scores.append(mos)
            else:
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
    parser.add_argument("--dataset", type=str, default="Scicom-intl/Evaluation-Multilingual-VC")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sampling", action="store_true", help="Whether to use sampling for generation")
    parser.add_argument("--sample_size", type=int, default=3, help="Number of samples to generate for each input when using sampling")
    args = parser.parse_args()

    main(args.dataset, 
         args.model_name, 
         args.output_dir, 
         args.length, 
         args.batch_size,
         args.sampling,
         args.sample_size)
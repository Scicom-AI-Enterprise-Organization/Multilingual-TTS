from datasets import (
    load_dataset, 
    DatasetDict, 
    Dataset as HFDataset
)
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor
)
from transformers.utils import logging as hf_logging
import xxhash
import argparse
import os
import torch
import re
import soundfile as sf
from tqdm import tqdm
import torchaudio.transforms as T
from jiwer import cer
import utmosv2 
from typing import Union, Literal
import json
import pandas as pd
import multiprocessing as mp
from functools import partial
import gc
import time
from torch.nn.utils.rnn import pad_sequence

hf_logging.set_verbosity_error()  # Suppress warnings from transformers library

class BaseTTSModel: 
    def __init__(self,
                 model_name: str, 
                 device: torch.device, 
                 sampling: bool = False, 
                 sample_size: int = 3, ):
        pass 
    
    @staticmethod
    def supported_languages() -> Union[list[str], None]:
        return None
    
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")
class ScicomTTSModel(BaseTTSModel): 
    def __init__(self, 
                 model_name: str, 
                 device: torch.device, 
                 sampling: bool = False, 
                 sample_size: int = 1, 
                 attn_implementation: str = "kernels-community/flash-attn3"):
        try: 
            from neucodec import NeuCodec
        except Exception as e: 
            raise ImportError("Failed to import NeuCodec. Please ensure neucodec is installed.") from e
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=attn_implementation, 
            torch_dtype=torch.bfloat16,
        ).to(device)
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
        output_tokens: torch.Tensor, # (1, S)
        tokenizer,
        codec,
        save_path: str,
    ): 
        "Decode output tokens to audio and save the audio file."
        if output_tokens.shape[0] > 1 and not isinstance(save_path, list) and len(save_path) != output_tokens.shape[0]:
            raise ValueError("save_path should be a list of paths with the same length as output_tokens batch size")
        
        decode_token = tokenizer.decode(output_tokens, skip_special_tokens=False)[0]
        codec_tokens = re.findall(r"<\|s_(\d+)\|>", decode_token)
        codec_tokens = torch.tensor([int(token) for token in codec_tokens])[None, None, :].to(codec.device)
        with torch.no_grad():
            audio = codec.decode_code(codec_tokens).to('cpu') # (1, 1, T)
            sf.write(save_path, audio[0,0].numpy(), samplerate=24000)
                
    def generate(self,  
                 target_text: str,
                 description: Union[str, None],
                 save_paths: str, 
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
            output_tokens = self.model.generate(
                **input_tokens, 
                **generation_kwargs
                )

        # decode output tokens to audio and save the audio file
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
                 sample_size: int = 3, 
                 attn_implementation: str = 'eager'): 
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError("QwenTTSModel requires the qwen-tts package. Please install it first.")
        self.model:Qwen3TTSModel = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation
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
    
    @staticmethod
    def supported_languages():
        return [
            'zh', 'en', 'fr', 'de', 'it', 'ja', 'ko', 'pt', 'ru', 'es'
        ]

    def generate(self,  
                 target_text: Union[str, list[str]],
                 description: Union[str, list[str], None],
                 save_paths: str, 
                 speaker_name: str = "Vivian",
                 **kwargs):
        
        lang = kwargs.get("language", "en")
        if lang not in self.supported_languages():
            return
        
        wav, sr = self.model.generate_custom_voice(
            text=target_text,
            language=self.mapped_language(lang),
            speaker=speaker_name,
            instruct=description
        )
        if sr != 24000:
            raise ValueError(f"Expected sample rate of 24000, but got {sr}. Resampling is needed")
        
        sf.write(save_paths, wav[0], samplerate=sr)
class ChatterBox(BaseTTSModel):
    def __init__(self, 
                 model_name: str, 
                 device: torch.device, 
                 sampling: bool = False, 
                 sample_size: int = 3, 
                 **kwargs):
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError:
            raise ImportError("ChatterBox model requires the chatterbox-tts package. Please install it first.")
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        self.sampling = sampling
        self.sample_size = sample_size
    
    @staticmethod
    def supported_languages():
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
                 target_text: str,
                 save_paths: str, 
                 description: Union[str, None] = None,
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

        try:
            wav = self.model.generate(target_text, language_id=lang)
            if self.model.sr != 24000:
                raise ValueError(f"Expected sample rate of 24000, but got {self.model_sr}. Resampling is needed")
            sf.write(save_paths, wav[0].cpu().numpy(), samplerate=self.model.sr)
        except Exception as e: 
            print(f"Error generating speech: {e} | Text: {target_text} | Language: {lang} | Output Path: {save_paths}")


MODEL_MAPPING = {
    "Scicom-intl/Multilingual-Expressive-TTS-1.7B" : ScicomTTSModel,
    "Scicom-intl/Multilingual-Expressive-TTS-0.6B" : ScicomTTSModel,
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": QwenTTSModel,
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": QwenTTSModel, 
    "chatterbox": ChatterBox,
}

class BaseDataset: 
    def __init__(self):
        assert isinstance(self.ds, HFDataset)
        assert "language" in self.ds.column_names
        assert "text" in self.ds.column_names
        assert "id" in self.ds.column_names
class Dataset(BaseDataset):
    def __init__(self, 
                 dataset_name: str, 
                 length: int = None, 
                 filter_langs = None):
        self.length = length
        self.ds = load_dataset(
            dataset_name, 
            "combine_filtered_whisper_large_v3", 
            split="train"
        )
        
        # transformation
        self.__filter_by_language()
        self.ds = self.ds.remove_columns(["source_text", "upvotes", "speaker_id", "audio_filename"])
        self.ds = self.ds.rename_column("target_text", "text")
        self.ds = self.ds.map(lambda sample: {"id": self.__hash_text(sample["text"])}, batched=False)
        if filter_langs is not None:
            self.ds = self.ds.filter(lambda sample: sample["language"] in filter_langs)
        
    @staticmethod
    def __hash_text(text: str)->str: 
        h64 = xxhash.xxh64()
        h64.update(text)
        return h64.hexdigest()
        
    def __get_language_mapping(self)->dict[str, str]:
        "Get the language code mapping to map with whisper supported languages."
        with open("common-voice-whisper-mapping.json", "r") as f:
            mapping = json.load(f)
        return mapping
    
    def __filter_by_language(self, lang_columns: str = "language"):
        df = self.ds.to_pandas()
        # map language to whisper supported language, else filter out the unsupported language
        mapping = self.__get_language_mapping()
        df[lang_columns] = df[lang_columns].map(mapping)
        num_dropped = df[lang_columns].isna().sum()
        if num_dropped > 0:
            print(f"Dropping {num_dropped} samples due to unsupported language.")
        else:
            print("All samples have supported languages.")
        df = df.dropna(subset=[lang_columns])
        self.ds = HFDataset.from_pandas(df)
    
    def split(
            self, 
            split_num: int, 
            prefix: Literal["trans", "mos", "output"], 
            sampling_size: int, 
            output_dir: str
        )->list[HFDataset]: 
        """Split the dataset into multiple subsets for parallel processing.
        
        Remove the samples that have already been processed.
        """
        consolidate_ds = []
        count = 0
        for sample in self.ds:
            lang = sample["language"]
            id = sample["id"]
            ext = "wav" if prefix == "output" else "txt"
            paths = [os.path.join(output_dir, f"{prefix}_{lang}_{id}_{i}.{ext}") for i in range(sampling_size)]
            count += 1
            if self.length is not None and count > self.length:
                break
            if all([os.path.exists(path) for path in paths]):
                continue
            consolidate_ds.append(sample)

        split_ds = []
        if len(consolidate_ds) == 0:
            return split_ds
        split_len = len(consolidate_ds) // split_num + (1 if len(consolidate_ds) % split_num > 0 else 0)
        for start in range(0, len(consolidate_ds), split_len):
            end = min(start + split_len, len(consolidate_ds))
            split_ds.append(
                HFDataset.from_list(consolidate_ds[start:end])
            )

        return split_ds

    def __iter__(self): 
        # merge all dataset 
        count = 0
        for sample in self.ds: 
            if self.length is not None and count >= self.length:
                return
            yield sample
            count += 1
    
    def __len__(self):
        return min(self.length, len(self.ds)) if self.length is not None else len(self.ds)
    
def repeat_interleave(lst: list[str], times: int)->list[str]:
    "Repeat each element in the list for a specified number of times."
    return [item for item in lst for _ in range(times)]

def tts(
    dataset: DatasetDict, 
    cuda_device: int,
    process_id: int,
    counter: mp.Value,
    model_name: str,
    output_dir: str, 
    sample_size: int = 1,
    batch_size: int = 1,
    attn_implementation: str = "kernels-community/flash-attn3",
):
    device = f"cuda:{cuda_device}"
    model = MODEL_MAPPING[model_name](model_name=model_name, device=device, attn_implementation=attn_implementation)
    
    with tqdm(total=len(dataset)*sample_size, desc=f"[PID{process_id}:Rank{cuda_device}] Generating TTS", position=process_id) as pbar:
        for start in range(0, len(dataset), 1):
            samples = dataset[start:start+1]
            save_paths = []
            for lang, id in zip(samples["language"], samples["id"]):
                save_paths.extend([os.path.join(output_dir, f"output_{lang}_{id}_{i}.wav") for i in range(sample_size)])

            # repeat interleave for sampling 
            target_texts = repeat_interleave(samples['text'], sample_size)
            descriptions = repeat_interleave(samples.get('APS', [None] * len(samples)), sample_size)
            languages = repeat_interleave(samples['language'], sample_size)
            
            for output_path, lang, target_text, description in zip(save_paths, languages, target_texts, descriptions):
                if os.path.exists(output_path):
                    pbar.update(1)
                    if counter is not None:
                        with counter.get_lock():
                            counter.value += 1
                    continue

                model.generate(
                    target_text=target_text,
                    description=description,
                    save_paths=output_path, 
                    language=lang
                )
                pbar.update(1)
                if counter is not None:
                    with counter.get_lock():
                        counter.value += 1

    # clear GPU memory and codec model
    model = None
    gc.collect()
    torch.cuda.empty_cache()


def stt(
    dataset: DatasetDict, 
    cuda_device: int,
    process_id: int,
    counter: mp.Value,
    output_dir: str, 
    sample_size: int = 1,
    batch_size: int = 1, 
    attn_implementation: str = "kernels-community/flash-attn3",
):
    device = f"cuda:{cuda_device}"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3", 
        torch_dtype=torch.bfloat16, # for backward compatibility
        attn_implementation=attn_implementation
    ).to(device)
    model = torch.compile(model)
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    sampler = T.Resample(orig_freq=24000, new_freq=16000)
    
    with tqdm(total=len(dataset)*sample_size, desc=f"[PID{process_id}:Rank{cuda_device}] Transcription", position=process_id) as pbar:
        batches_input = []
        batches_language = None
        batches_output_paths = []

        for start in range(0, len(dataset)):
            samples = dataset[start:start+1]
            save_paths = []
            audio_paths = []
            for lang, id in zip(samples["language"], samples["id"]):
                save_paths.extend([os.path.join(output_dir, f"trans_{lang}_{id}_{i}.txt") for i in range(sample_size)])
                audio_paths.extend([os.path.join(output_dir, f"output_{lang}_{id}_{i}.wav") for i in range(sample_size)])

            # load audio and preprocess
            languages = repeat_interleave(samples['language'], sample_size)
            for audio_path, save_path, language in zip(audio_paths, save_paths, languages):
                try:
                    audio, sr = sf.read(audio_path)
                    audio = torch.from_numpy(audio).to(torch.float32)
                
                    audio = sampler(audio) # (T')

                    batches_input.append(audio)
                    batches_output_paths.append(save_path)
                    batches_language = language
                    next_lang = dataset[start+1]['language'] if start+1 < len(dataset) else None
                    if len(batches_input) < batch_size and next_lang == batches_language:
                        continue
                    
                    inputs = processor(
                        pad_sequence(batches_input, batch_first=True).numpy(), # (B, T') 
                        sampling_rate=16000, 
                        return_tensors="pt", 
                    ).to(device)

                    with torch.no_grad(), torch.autocast(device_type=device):
                        predicted_ids = model.generate(**inputs, language=language, task="transcribe")
                    
                    transcriptions = processor.batch_decode(predicted_ids.cpu().numpy(), skip_special_tokens=True)
                    
                    for transcription, save_path in zip(transcriptions, batches_output_paths):
                        with open(save_path, "w") as f:
                            f.write(transcription.strip())
                except Exception as e: 
                    print(f"Error processing audio: {e} | Audio path: {audio_path}")
                pbar.update(len(batches_input))
                if counter is not None:
                    with counter.get_lock():
                        counter.value += len(batches_input)
                batches_input = []
                batches_output_paths = []
                batches_language = None
    
    model = None 
    gc.collect()
    torch.cuda.empty_cache() 


def evaluate_mos(
    dataset: DatasetDict, 
    cuda_device: int,
    process_id: int,
    counter: mp.Value,
    output_dir: str, 
    sample_size: int = 1, 
    batch_size: int = 1,
    attn_implementation: str = "kernels-community/flash-attn3",
):
    device = f"cuda:{cuda_device}"
    model = utmosv2.create_model(pretrained=True, device=device)

    with tqdm(total=len(dataset)*sample_size, desc=f"[PID{process_id}:Rank{cuda_device}] MOS evaluation", position=process_id) as pbar:
        for start in range(0, len(dataset)):
            samples = dataset[start:start+1]
            save_paths = []
            audio_paths = []
            for lang, id in zip(samples["language"], samples["id"]):
                save_paths.extend([os.path.join(output_dir, f"mos_{lang}_{id}_{i}.txt") for i in range(sample_size)])
                audio_paths.extend([ os.path.join(output_dir, f"output_{lang}_{id}_{i}.wav") for i in range(sample_size)])

            for audio_path, save_path in zip(audio_paths, save_paths):
                if os.path.exists(save_path):
                    pbar.update(1)
                    with counter.get_lock():
                        counter.value += 1
                    continue
                
                try:
                    waveform, sr = sf.read(audio_path)
                    mos = model.predict(data=torch.tensor(waveform, dtype=torch.float32), 
                                        sr=sr,
                                        num_workers=0, 
                                        verbose=False).numpy()[0]
                    with open(save_path, "w") as f:
                        f.write(f"{mos:.5f}")
                except Exception as e:
                    print(f"Error processing audio: {e} | Audio path: {audio_path}")
                pbar.update(1)
                with counter.get_lock():
                    counter.value += 1

    model = None 
    gc.collect()
    torch.cuda.empty_cache()

def track_progress(counter: mp.Value, total: int, position: int):
    with tqdm(total=total, desc="Overall Progress", position=position, unit="sample", smoothing=0) as pbar:
        while True:
            with counter.get_lock():
                current = counter.value
            pbar.update(current - pbar.n)
            if current >= total:
                break
            time.sleep(1)

def parallel_eval(
    dataset: Dataset, 
    devices: list[int],
    proc: Literal["tts", "stt", "mos"], 
    sample_size: int, 
    output_dir: str, 
    batch_size: int = 1,
    model_name: str = None,
    attn_implementation: str = "kernels-community/flash-attn3",
):
    """
    Parallel evaluation for TTS generation, transcription for CER, and MOS evaluation.
    """
    _PREFIX_MAP = {
        "tts": "output",
        "stt": "trans", 
        "mos": "mos",
    }
    _FUNC_MAP = {
        "tts": tts,
        "stt": stt, 
        "mos": evaluate_mos,
    }
    ds = dataset.split(
        split_num=len(devices) or 1,
        prefix=_PREFIX_MAP[proc],
        sampling_size=sample_size,
        output_dir=output_dir
    )
    total_samples = sum([len(_ds) for _ds in ds]) * sample_size
    if len(ds) == 0:
        print(f"No samples to process for {proc.upper()}. All done!")
        return
    
    run_func_kwargs = {
        "output_dir": output_dir,
        "sample_size": sample_size,
        "attn_implementation": attn_implementation,
        "batch_size": batch_size,
    }
    if proc == "tts":
        run_func_kwargs["model_name"] = model_name
    run_func = partial(
        _FUNC_MAP[proc],
        **run_func_kwargs
    )
    
    processes = []
    ctx = mp.get_context("spawn")
    counter = ctx.Value('i', 0)
    for i, _ds in enumerate(ds):
        p = ctx.Process(target=run_func, args=(_ds, devices[i], i, counter))
        processes.append(p)
        p.start()
    
    p_track = ctx.Process(target=track_progress, args=(counter, total_samples, len(devices)))
    p_track.start()
    processes.append(p_track)
    
    for p in processes:
        p.join()

def summarize(dataset: Dataset, output_dir: str, sample_size: int, skip_mos: bool):
    summary = {}
    with tqdm(total=len(dataset)*sample_size, desc="Calculating CER/MOS", position=0) as pbar:
        for sample in dataset:
            lang = sample["language"]
            if lang not in summary:
                summary[lang] = {
                    "cer": [],
                    "mos": []
                }
                
            for i in range(sample_size):
                trans_path = os.path.join(output_dir, f"trans_{lang}_{sample['id']}_{i}.txt")
                
                try:
                    if os.path.exists(trans_path):
                        with open(trans_path, "r") as f:
                            transcription = f.read().strip()
                        cer_score = min(cer(sample["text"], transcription), 1.0)
                        summary[lang]["cer"].append(cer_score)
                except:
                    raise ValueError(f"Content of transcription file is invalid: {trans_path}")
                
                mos_path = os.path.join(output_dir, f"mos_{lang}_{sample['id']}_{i}.txt")
                try:
                    if os.path.exists(mos_path) and not skip_mos:
                        with open(mos_path, "r") as f:
                            mos = float(f.read().strip())
                        summary[lang]["mos"].append(mos)
                except: 
                    raise ValueError(f"Content of mos file is invalid: {mos_path}")
                pbar.update(1)

    print("\nEvaluation Summary:")
    summary_df = pd.DataFrame({
        lang: {
            "average_cer": sum(metrics["cer"]) / len(metrics["cer"]) if len(metrics["cer"]) > 0 else None,
            "average_mos": sum(metrics["mos"]) / len(metrics["mos"]) if len(metrics["mos"]) > 0 else None,
        } for lang, metrics in summary.items()
    }).T
    print(summary_df.sort_values(by="average_cer"))
    summary_df.sort_values(by="average_cer").to_csv(os.path.join(output_dir, f"eval_summary.csv"), sep="\t")
    return summary_df

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Scicom-intl/Multilingual-Expressive-TTS-1.7B")
    parser.add_argument("--dataset", type=str, default="Scicom-intl/Evaluation-Multilingual-VC")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save generated audio")
    parser.add_argument("--length", type=int, default=None, help="Subset of dataset for dry run")
    parser.add_argument("--sample_size", type=int, default=3, help="Number of samples to generate for each input")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--skip_mos", action="store_true", help="Whether to skip MOS evaluation")
    parser.add_argument("--skip_tts", action="store_true", help="Whether to skip TTS evaluation")
    parser.add_argument("--skip_stt", action="store_true", help="Whether to skip STT evaluation")
    parser.add_argument("--attn_implementation", type=str, default="kernels-community/flash-attn3", help="Attention implementation for TTS model")
    parser.add_argument("--replicate", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.model_name in MODEL_MAPPING:
        raise ValueError(f"Model {args.model_name} is not supported. Please choose from {list(MODEL_MAPPING.keys())}")

    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    devices = devices * args.replicate
    process_id = list(range(len(devices)))

    filter_langs = MODEL_MAPPING[args.model_name].supported_languages()
    dataset = Dataset(args.dataset, length=args.length, filter_langs=filter_langs)
    print(f"Using devices {devices} for evaluation with model {args.model_name} on dataset {args.dataset} | total sample: {len(dataset)}")

    # TODO: auto calculate the replicate (but only you got the time, not important)
    # 1. Run TTS
    if not args.skip_tts:
        parallel_eval(
            dataset=dataset,
            devices=devices,
            proc="tts",
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            model_name=args.model_name,
            attn_implementation=args.attn_implementation,
        )

    # 2. Run STT
    if not args.skip_stt:
        parallel_eval(
            dataset=dataset,
            devices=devices,
            proc="stt",
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            attn_implementation=args.attn_implementation,
        )

    # 3. Run MOS evaluation
    if not args.skip_mos:
        parallel_eval(
            dataset=dataset,
            devices=devices,
            proc="mos",
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            attn_implementation=args.attn_implementation,
        )

    # 4. Summarize results
    summarize(
        dataset, 
        args.output_dir, 
        args.sample_size, 
        args.skip_mos
    )

# Multilingual-TTS Evaluation

## Evaluation Metrics
1. MOS (UTMOSv2)
2. CER (Whisper Large v3)

## Benchmark Dataset 
1. ~[InstructTTSEval](https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval/viewer/default/zh) (Langauges: EN, ZH)~
2. [Scicom-intl/Evaluation-Multilingual-VC](https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC)

## Benchmark Output 
### Model: scicom-multilingual-expressive-tts.1.7B
Generation Configuration
- Speaker: genshin-voice_audio_Rahman
- Temperature: 0.8
- Repetition Penalty: 1.15

Command: 
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 --allow-existing
source .venv/bin/activate

uv pip install datasets transformers soundfile jiwer kernels xxhash  
uv pip install neucodec
uv pip install git+https://github.com/sarulab-speech/UTMOSv2.git

hf download Scicom-intl/Evaluation-Multilingual-VC --exclude "*.zip" --repo-type dataset

python evaluate.py \
    --model_name Scicom-intl/Multilingual-Expressive-TTS-1.7B \
    --sample_size 3 \
    --output_dir ./scicom-multilingual-expressive-tts-1.7B \
    --attn_implementation kernels-community/flash-attn3 \
    --replicate 4
```


Result:
| lang | average_cer | average_mos |
|------|------------:|------------:|
| en   | 0.0237 | 3.2345 |
| de   | 0.0544 | 2.9225 |
| ru   | 0.0612 | 2.6147 |
| id   | 0.0645 | 2.6716 |
| ko   | 0.0919 | 2.8312 |
| fr   | 0.1010 | 2.7933 |
| es   | 0.1024 | 2.9424 |
| uk   | 0.1120 | 2.5321 |
| mk   | 0.1189 | 2.6371 |
| be   | 0.1202 | 2.7241 |
| it   | 0.1444 | 2.9254 |
| hy   | 0.1494 | 2.6320 |
| ka   | 0.1533 | 2.7533 |
| nl   | 0.1551 | 2.9618 |
| bg   | 0.1572 | 2.4995 |
| ta   | 0.1650 | 2.6946 |
| ur   | 0.1679 | 2.5551 |
| gl   | 0.1683 | 2.9318 |
| el   | 0.1862 | 2.4936 |
| eu   | 0.1865 | 2.9735 |
| hi   | 0.1927 | 2.7140 |
| lt   | 0.2053 | 2.9330 |
| ca   | 0.2055 | 2.8928 |
| sl   | 0.2067 | 2.8544 |
| sw   | 0.2126 | 2.9751 |
| az   | 0.2187 | 2.7124 |
| pt   | 0.2227 | 2.7894 |
| ro   | 0.2243 | 2.8396 |
| af   | 0.2249 | 2.9168 |
| tr   | 0.2317 | 2.6099 |
| kk   | 0.2326 | 2.5081 |
| fa   | 0.2393 | 2.5521 |
| cs   | 0.2412 | 2.8382 |
| sv   | 0.2421 | 2.7235 |
| ha   | 0.2472 | 3.0032 |
| mr   | 0.2508 | 2.7432 |
| et   | 0.2544 | 2.9513 |
| lv   | 0.2646 | 2.8369 |
| ja   | 0.2652 | 2.7130 |
| uz   | 0.2818 | 3.0098 |
| sq   | 0.2891 | 2.7835 |
| ar   | 0.2982 | 2.5598 |
| fi   | 0.3066 | 2.7859 |
| hu   | 0.3151 | 2.9497 |
| tg   | 0.3163 | 2.6276 |
| da   | 0.3241 | 2.7600 |
| pl   | 0.3275 | 2.6897 |
| nn   | 0.3317 | 2.7708 |
| he   | 0.3488 | 2.5346 |
| ne   | 0.3514 | 2.5835 |
| tt   | 0.3591 | 2.4610 |
| oc   | 0.3711 | 2.7406 |
| mt   | 0.3758 | 2.8066 |
| zh   | 0.3770 | 2.8151 |
| th   | 0.3925 | 2.4411 |
| vi   | 0.4130 | 2.5196 |
| mn   | 0.4148 | 2.4659 |
| sk   | 0.4157 | 2.6535 |
| cy   | 0.4295 | 2.7419 |
| ps   | 0.4396 | 2.4289 |
| is   | 0.4402 | 2.8253 |
| br   | 0.4423 | 2.6744 |
| ht   | 0.4626 | 2.5975 |
| yi   | 0.4832 | 2.5874 |
| bn   | 0.4950 | 2.3717 |
| pa   | 0.4991 | 2.8267 |
| tk   | 0.5161 | 2.7754 |
| yo   | 0.5365 | 2.9139 |
| sr   | 0.6407 | 2.3878 |
| yue  | 0.7963 | 2.7234 |
| ba   | 0.8114 | 2.4282 |
| te   | 0.8635 | 2.6563 |
| as   | 0.9615 | 2.4714 |
| sd   | 0.9621 | 2.4282 |
| ml   | 0.9695 | 2.7801 |
| lo   | 0.9996 | 2.2825 |
| am   | 1.0000 | 2.7324 |


### Model: Qwen3-TTS-12Hz-1.7B-CustomVoice
Generation Configuration
- Speaker: Vivian
- Temperature: 0.8
- Repetition Penalty: 1.05

Command:
```shell
uv venv --python 3.12 --allow-existing
source .venv/bin/activate

uv pip install datasets transformers soundfile jiwer kernels xxhash  
uv pip install git+https://github.com/sarulab-speech/UTMOSv2.git

# Download Dataset
hf download Scicom-intl/Evaluation-Multilingual-VC --exclude "*.zip" --repo-type dataset

# Run eval
python evaluate.py \
    --model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --output_dir ./qwen-tts \
    --attn_implementation kernels-community/flash-attn3 \
    --sample_size 3 \
    --replicate 4
```

Result: 
| language | average_cer | average_mos |
| -------- | ----------: | ----------: |
| de       |      0.0252 |      2.9004 |
| en       |      0.0294 |      3.1397 |
| es       |      0.0369 |      2.7867 |
| it       |      0.0431 |      2.5595 |
| ru       |      0.0436 |      2.4544 |
| fr       |      0.0578 |      2.7012 |
| ko       |      0.0604 |      2.5202 |
| pt       |      0.1284 |      2.6750 |
| ja       |      0.2228 |      2.6036 |
| zh       |      0.2741 |      2.7607 |


### Model: Chatterbox
Command: 
```shell
uv venv --python 3.11 --allow-existing # only support 3.11 and below for chatterbox
source .venv/bin/activate

uv pip install setuptools git+https://github.com/resemble-ai/chatterbox.git git+https://github.com/resemble-ai/Perth.git
uv pip install datasets soundfile jiwer kernels xxhash  
uv pip install git+https://github.com/sarulab-speech/UTMOSv2.git

hf download Scicom-intl/Evaluation-Multilingual-VC --exclude "*.zip" --repo-type dataset

python evaluate.py \
    --model_name chatterbox \
    --output_dir ./chatterbox-tts \
    --sample_size 3
```

Result: 
On-going
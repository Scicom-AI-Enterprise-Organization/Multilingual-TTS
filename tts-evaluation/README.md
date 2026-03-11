# Multilingual-TTS Evaluation

## Evaluation Metrics
1. MOS (UTMOSv2)
2. CER (Whisper Large v3)

## Benchmark Dataset 
1. ~[InstructTTSEval](https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval/viewer/default/zh) (Langauges: EN, ZH)~
2. [Scicom-intl/Evaluation-Multilingual-VC](https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC)

## How to run? 
Setup
```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install vllm datasets transformers torch soundfile jiwer kernels xxhash
uv pip install neucodec
uv pip install git+https://github.com/sarulab-speech/UTMOSv2.git
```

Benchmark
```bash
# evaluate TTS model with pass@k, k=3
python evaluate.py \
    --model_name Scicom-intl/Multilingual-Expressive-TTS-1.7B \
    --sampling \
    --sample_size 3
```

## Benchmark Output 
### Model: scicom-multilingual-expressive-tts.1.7B
Generation Configuration
- Speaker: genshin-voice_audio_Rahman
- Temperature: 0.8
- Repetition Penalty: 1.15

| lang | average_cer | average_mos |
|-----|-------------|-------------|
| en | 0.0237 | - |
| de | 0.0691 | - |
| ko | 0.0972 | - |
| fr | 0.1188 | - |
| ru | 0.1313 | - |
| es | 0.1464 | - |
| id | 0.1812 | - |
| it | 0.2216 | - |
| ja | 0.2704 | - |
| gl | 0.2844 | - |
| pt | 0.3209 | - |
| ha | 0.3596 | - |
| be | 0.3598 | - |
| ca | 0.3672 | - |
| eu | 0.3703 | - |
| ta | 0.3707 | - |
| sw | 0.3842 | - |
| zh | 0.3866 | - |
| uz | 0.3919 | - |
| lt | 0.3995 | - |
| az | 0.4060 | - |
| ro | 0.4082 | - |
| hy | 0.4190 | - |
| uk | 0.4225 | - |
| tr | 0.4348 | - |
| bg | 0.4428 | - |
| et | 0.4440 | - |
| sl | 0.4517 | - |
| nl | 0.4564 | - |
| af | 0.4596 | - |
| el | 0.4674 | - |
| oc | 0.4683 | - |
| hi | 0.4685 | - |
| sv | 0.4715 | - |
| lv | 0.4769 | - |
| sq | 0.4825 | - |
| hu | 0.4868 | - |
| fi | 0.4997 | - |
| mt | 0.5008 | - |
| da | 0.5025 | - |
| nn | 0.5032 | - |
| is | 0.5036 | - |
| ka | 0.5140 | - |
| cs | 0.5179 | - |
| pl | 0.5406 | - |
| br | 0.5411 | - |
| ar | 0.5501 | - |
| sk | 0.5625 | - |
| tk | 0.5628 | - |
| cy | 0.5654 | - |
| mk | 0.5733 | - |
| he | 0.6136 | - |
| yo | 0.6346 | - |
| mr | 0.6375 | - |
| ht | 0.6718 | - |
| fa | 0.7050 | - |
| sr | 0.7317 | - |
| yue | 0.7736 | - |
| ne | 0.7832 | - |
| tg | 0.8176 | - |
| th | 0.8177 | - |
| bn | 0.8226 | - |
| pa | 0.8241 | - |
| vi | 0.8281 | - |
| ur | 0.8378 | - |
| kk | 0.8398 | - |
| mn | 0.8402 | - |
| tt | 0.8525 | - |
| ba | 0.8574 | - |
| ps | 0.8767 | - |
| sd | 0.8793 | - |
| as | 0.8861 | - |
| te | 0.9117 | - |
| yi | 0.9437 | - |
| ml | 0.9617 | - |
| am | 0.9640 | - |
| lo | 0.9868 | - |
# Multilingual-TTS Evaluation

## Evaluation Metrics
1. MOS (UTMOSv2)
2. CER (Whisper Large v3)

## Benchmark Dataset 
1. [InstructTTSEval](https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval/viewer/default/zh) (Langauges: EN, ZH)

## How to run? 
Setup
```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install vllm datasets transformers torch soundfile jiwer
uv pip install neucodec
uv pip install git+https://github.com/sarulab-speech/UTMOSv2.git
```

Benchmark
```bash
# evaluate TTS model with pass@k, k=3
python evaluate.py \
    --model_name Scicom-intl/Multilingual-Expressive-TTS-1.7B \
    --batch_size 10 \
    --sampling 
    --sample_size 3
```

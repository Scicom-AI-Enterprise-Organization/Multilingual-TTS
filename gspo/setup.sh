# uv venv --python 3.12
# source venv/bin/activate

# Install dependencies
uv pip install transformers soundfile neucodec ipykernel
uv pip install nemo_toolkit[all]
python -m ipykernel install --user --name=grpo
git pull
pip install -e .
pip install git+https://github.com/huggingface/accelerate

rm -f ../.cache
huggingface-cli login

python examples/get_started.py
pip install pytorch-lightning==1.8.1
pip install --upgrade git+https://github.com/huggingface/transformers.git
pip install datasets[audio] evaluate jiwer
pip install accelerate -U

git clone https://github.com/facebookresearch/fairseq
cd fairseq
pip install --editable ./

pip install editdistance soundfile
pip install wandb

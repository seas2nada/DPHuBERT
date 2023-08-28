pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pytorch-lightning==1.8.1
pip install transformers==4.24.0

git clone https://github.com/facebookresearch/fairseq
cd fairseq
pip install --editable ./

pip install editdistance soundfile

import editdistance
a = "/home/ubuntu/Workspace/DB/LibriSpeech/DPHuBERT/exp/whisper-large-v3/cv16_ko-decoding_log.txt"

with open(a, 'r') as f:
    lines = f.readlines()

# Prediction: 가리등에 비치어 떨어지는 눈송이는 마치 여름날 전등불을 싸고 날아드는 하루살이 때 같았다
# Reference: 가로등에 비치어 떨어지는 눈송이는 마치 여름날 전등불을 싸고 날아드는 하루살이떼 같았다.

pred = []
ref = []
for line in lines:
    if "Prediction" in line:
        pred.append(line.split(":")[1].strip())
    if "Reference" in line:
        ref.append(line.split(":")[1].strip())

wrongs = 0
total_len = 0
for p, r in zip(pred, ref):
    wrongs += editdistance.eval(p.split(" "), r.split(" "))
    total_len += len(r.split(" "))

print(wrongs / total_len)
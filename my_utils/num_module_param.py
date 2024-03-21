with open("whisper_num_params.txt", 'r') as f:
    lines = f.readlines()

dic = {}
for line in lines:
    line = line.rstrip('\n')
    key = line.split(": ")[0]
    value = line.split(": ")[1]
    dic[key] = value

enc, dec, rest = 0, 0, 0
for key, value in dic.items():
    value = int(value)
    if "model.encoder" in key:
        enc += value
    if "model.decoder" in key:
        dec += value
    else:
        rest += value

print(enc)
print(dec)

print(enc+dec+value)
import os

subsets = ["dev_clean", "dev_other", "test_clean", "test_other", "ted-valid", "ted-test", "cv-dev", "cv-test", "l2arc-valid", "dt05", "et05"]
dir = "/home/ubuntu/Workspace/DB/LibriSpeech/DPHuBERT/data"

num_words = {}
for subset in subsets:
    num_words[subset] = 0

for subset in subsets:
    if subset in ["dev_clean", "dev_other", "test_clean", "test_other"]:
        subsubset = "librispeech"
    elif subset in ["ted-valid", "ted-test"]:
        subsubset = "TED"
    elif subset in ["cv-dev", "cv-test"]:
        subsubset = "commonvoice"
    elif subset in ["l2arc-valid"]:
        subsubset = "l2arctic"
    elif subset in ["dt05", "et05"]:
        subsubset = "chime3"

    with open(os.path.join(dir, subsubset, subset, subset + ".wrd"), "r") as f:
        lines = f.readlines()
    # Count the number of words in the subset
    for line in lines:
        words = line.strip().split()
        num_words[subset] += len(words)

with open("num_of_words.txt", "w") as f:
    for k, v in num_words.items():
        f.write(f"{k}: {v}\n")
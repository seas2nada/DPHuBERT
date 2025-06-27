from pathlib import Path
import argparse
import tensorflow as tf   # works with TF 2.x
from tensorflow.python.summary.summary_iterator import summary_iterator

def main():
    parser = argparse.ArgumentParser("Check log")
    parser.add_argument(
        "--log",
        required=True,
    )

    args = parser.parse_args()
    path = args.log

    for e in summary_iterator(path):
        step = e.step
        wall_time = e.wall_time
        for v in e.summary.value:          # there can be multiple values per step
            print(step, v.tag, v.simple_value)

if __name__ == "__main__":
    main()
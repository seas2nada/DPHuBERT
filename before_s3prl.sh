OUTPUT_DIR="acphubert"  # change

rm -rf s3prl/s3prl/upstream/$OUTPUT_DIR; cp -r $OUTPUT_DIR s3prl/s3prl/upstream/$OUTPUT_DIR
rm -rf tools/venv/lib/python3.10/site-packages/s3prl/upstream/$OUTPUT_DIR; cp -r $OUTPUT_DIR tools/venv/lib/python3.10/site-packages/s3prl/upstream/$OUTPUT_DIR

# add upstream importing line in <s3prl root>/s3prl/hub.py
# from s3prl.upstream.[model_name].hubconf import *
# Ex) from s3prl.upstream.armhubert.hubconf import *

rm -rf tools/venv/lib/python3.10/site-packages/s3prl/upstream tools/venv/lib/python3.10/site-packages/s3prl/hub.py
cp -r s3prl/s3prl/upstream tools/venv/lib/python3.10/site-packages/s3prl/
cp -r s3prl/s3prl/hub.py tools/venv/lib/python3.10/site-packages/s3prl/hub.py
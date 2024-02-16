import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt

INPUT_DIR="/home/sitb157/datas/saved_model"
OUTPUT_DIR="/home/sitb157/datas/trt_saved_model"

converter = trt.TrtGraphConverterV2(input_saved_model_dir=INPUT_DIR)
converter.convert()
converter.save(OUTPUT_DIR)

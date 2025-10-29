Run train and quantize outside of vitis-ai container in python virtual env
python3 -m venv venv
source venv/bin/activate

pip install torch torchvision tqdm
pip install "timm==0.6.13"

Once you have a quantized .xmodel
Launch vitis-ai conatainer:
docker run -v .:/workspace -it xilinx/vitis-ai /bin/bash

run compiler.sh
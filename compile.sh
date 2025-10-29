#!/bin/sh

# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Mark Harvey, Xilinx Inc

ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
TARGET=KV260
echo "-----------------------------------------"
echo "COMPILING MODEL FOR $TARGET.."
echo "-----------------------------------------"

BUILD=$1
LOG=$2

mkdir -p ${LOG}

compile() {
  vai_c_xir \
  --xmodel      ${BUILD}/quant_model/ResNet_int.xmodel \
  --arch        $ARCH \
  --net_name    CNN_${TARGET} \
  --output_dir  ${BUILD}/compiled_model
}

compile 2>&1 | tee ${LOG}/compile_$TARGET.log

echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"




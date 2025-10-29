#!/bin/sh


ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
TARGET=KV260
echo "-----------------------------------------"
echo "COMPILING MODEL FOR $TARGET.."
echo "-----------------------------------------"
xmodel=*.xmodel


compile() {
  vai_c_xir \
  --xmodel      build/quantModel/soilQualityClassifier_int.xmodel \
  --arch        $ARCH \
  --net_name    soildQualityClassifier_KV260 \
  --output_dir  build/compiledXmodel
}

compile 2>&1 

echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"




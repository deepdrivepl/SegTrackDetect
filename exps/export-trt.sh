/workspace/tensorrt/bin/trtexec \
--onnx=/onnx/006-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-300ep-best.onnx \
--minShapes=images:1x3x512x512 \
--optShapes=images:8x3x512x512 \
--maxShapes=images:64x3x512x512 \
--saveEngine=/onnx/SDS-v7t-006-fp32.engine \
--verbose

/workspace/tensorrt/bin/trtexec \
--onnx=/onnx/SeaDronesSee-001-R18-448x768-best-loss.onnx \
--minShapes=inputs:1x3x448x768 \
--optShapes=inputs:1x3x448x768 \
--maxShapes=inputs:64x3x448x768 \
--saveEngine=/onnx/SDS-unetR18-001-fp32.engine \
--verbose

/workspace/tensorrt/bin/trtexec \
--onnx=/onnx/006-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-300ep-best.onnx \
--minShapes=images:1x3x512x512 \
--optShapes=images:8x3x512x512 \
--maxShapes=images:64x3x512x512 \
--saveEngine=/onnx/SDS-v7t-006-fp16.engine \
--verbose --fp16

/workspace/tensorrt/bin/trtexec \
--onnx=/onnx/SeaDronesSee-001-R18-448x768-best-loss.onnx \
--minShapes=inputs:1x3x448x768 \
--optShapes=inputs:1x3x448x768 \
--maxShapes=inputs:64x3x448x768 \
--saveEngine=/onnx/SDS-unetR18-001-fp16.engine \
--verbose --fp16


/workspace/tensorrt/bin/trtexec \
--onnx=/onnx/006-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-300ep-best.onnx \
--minShapes=images:1x3x512x512 \
--optShapes=images:8x3x512x512 \
--maxShapes=images:64x3x512x512 \
--saveEngine=/onnx/SDS-v7t-006-int8.engine \
--verbose --int8

/workspace/tensorrt/bin/trtexec \
--onnx=/onnx/SeaDronesSee-001-R18-448x768-best-loss.onnx \
--minShapes=inputs:1x3x448x768 \
--optShapes=inputs:1x3x448x768 \
--maxShapes=inputs:64x3x448x768 \
--saveEngine=/onnx/SDS-unetR18-001-int8.engine \
--verbose --int8

# 004-DroneCrowd-yolov7-tiny-512x512-crops-only-multiple-scales-50ep-best.onnx

# DroneCrowd-001-R18-192x320-best-loss.onnx


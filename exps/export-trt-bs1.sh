/workspace/tensorrt/bin/trtexec \
--onnx=/tinyROI/onnx-bs1/006-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-300ep-best.onnx \
--saveEngine=/tinyROI/onnx-bs1/SDS-v7t-006-fp32.engine \
--verbose

/workspace/tensorrt/bin/trtexec \
--onnx=/tinyROI/onnx-bs1/SeaDronesSee-001-R18-448x768-best-loss.onnx \
--saveEngine=/tinyROI/onnx-bs1/SDS-unetR18-001-fp32.engine \
--verbose

/workspace/tensorrt/bin/trtexec \
--onnx=/tinyROI/onnx-bs1/006-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-300ep-best.onnx \
--saveEngine=/tinyROI/onnx-bs1/SDS-v7t-006-fp16.engine \
--verbose --fp16

/workspace/tensorrt/bin/trtexec \
--onnx=/tinyROI/onnx-bs1/SeaDronesSee-001-R18-448x768-best-loss.onnx \
--saveEngine=/tinyROI/onnx-bs1/SDS-unetR18-001-fp16.engine \
--verbose --fp16


/workspace/tensorrt/bin/trtexec \
--onnx=/tinyROI/onnx-bs1/006-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-300ep-best.onnx \
--saveEngine=/tinyROI/onnx-bs1/SDS-v7t-006-int8.engine \
--verbose --int8

/workspace/tensorrt/bin/trtexec \
--onnx=/tinyROI/onnx-bs1/SeaDronesSee-001-R18-448x768-best-loss.onnx \
--saveEngine=/tinyROI/onnx-bs1/SDS-unetR18-001-int8.engine \
--verbose --int8

# 004-DroneCrowd-yolov7-tiny-512x512-crops-only-multiple-scales-50ep-best.onnx

# DroneCrowd-001-R18-192x320-best-loss.onnx


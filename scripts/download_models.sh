OUT_DIR="weights"

mkdir -p $OUT_DIR

echo "Downloading models"
echo "ROI Estimation Models"
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/DroneCrowd-001-R18-192x320-best-loss.pt" -P $OUT_DIR
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/DroneCrowd-001-R18-384x640-best-loss.pt" -P $OUT_DIR
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/DroneCrowd-001-R18-96x160-best-loss.pt" -P $OUT_DIR
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/SeaDronesSee-000-R18-128x192-best-loss.pt" -P $OUT_DIR
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/SeaDronesSee-000-R18-224x384-best-loss.pt" -P $OUT_DIR
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/SeaDronesSee-000-R18-448x768-best-loss.pt" -P $OUT_DIR
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/SeaDronesSee-000-R18-64x96-best-loss.pt" -P $OUT_DIR
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/unetR18-ZebraFish.pt" -P $OUT_DIR
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/u2netp_MTSD.pt" -P $OUT_DIR


echo "Detection Models"
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/004-DroneCrowd-yolov7-tiny-512x512-crops-only-multiple-scales-50ep-best.torchscript.pt" -P $OUT_DIR
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/006-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-300ep-best.torchscript.pt" -P $OUT_DIR
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/yolov7t-ZebraFish.pt" -P $OUT_DIR
wget -nv "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/yolov4_MTSD.pt" -P $OUT_DIR
OUT_DIR="/tinyROI/weights"

mkdir -p $OUT_DIR

echo "Downloading models"
wget "https://github.com/koseq/tinyROI-track/releases/download/v0.1/yolov7t-ZebraFish.pt" -P $OUT_DIR
wget "https://github.com/koseq/tinyROI-track/releases/download/v0.1/unetR18-ZebraFish.pt" -P $OUT_DIR
wget "https://github.com/koseq/tinyROI-track/releases/download/v0.1/yolov4_MTSD.pt" -P $OUT_DIR
wget "https://github.com/koseq/tinyROI-track/releases/download/v0.1/u2netp_MTSD.pt" -P $OUT_DIR
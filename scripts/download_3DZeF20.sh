OUT_DIR="/tinyROI/data"

mkdir -p $OUT_DIR

echo "Downloading 3DZeF20 dataset"
wget "https://motchallenge.net/data/3DZeF20.zip" -O $OUT_DIR/3DZeF20.zip

unzip $OUT_DIR/3DZeF20.zip -d $OUT_DIR
rm $OUT_DIR/3DZeF20.zip

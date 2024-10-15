OUT_DIR="data/SeaDronesSee"

mkdir -p $OUT_DIR

echo "Downloading SeaDronesSee MOT dataset"
wget "https://cloud.cs.uni-tuebingen.de/index.php/s/W4ztazMxqfHdYWA/download?path=%2Fannotations&files=instances_train_objects_in_water.json" -O $OUT_DIR/instances_train_objects_in_water.json
wget "https://cloud.cs.uni-tuebingen.de/index.php/s/W4ztazMxqfHdYWA/download?path=%2Fannotations&files=instances_val_objects_in_water.json" -O $OUT_DIR/instances_val_objects_in_water.json
wget "https://cloud.cs.uni-tuebingen.de/index.php/s/W4ztazMxqfHdYWA/download?path=%2Fannotations&files=instances_test_objects_in_water.json" -O $OUT_DIR/instances_test_objects_in_water.json
wget "https://cloud.cs.uni-tuebingen.de/index.php/s/W4ztazMxqfHdYWA/download?path=%2F&files=SeaDronesSee_MOT_jpg_compressed.zip&downloadStartSecret=2aqth6llkvr" -O $OUT_DIR/images.zip

unzip $OUT_DIR/images.zip -d $OUT_DIR
mv "$OUT_DIR/Compressed" "$OUT_DIR/images" 
rm $OUT_DIR/images.zip
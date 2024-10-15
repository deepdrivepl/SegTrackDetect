OUT_DIR="data/MTSD"

mkdir -p $OUT_DIR

echo "Downloading the old MTSD annotation version"
echo ""
wget "https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/annotations_v1.zip" -O $OUT_DIR/annotations_v1.zip
unzip $OUT_DIR/annotations_v1.zip -d $OUT_DIR
rm $OUT_DIR/annotations_v1.zip

echo -e "\nVisit 'https://www.mapillary.com/dataset/trafficsign',\ncreate an account,\nand download the MTSD dataset yourself"
echo ""
echo "The data structure should be as follows:
MTSD/
├── annotations
├── annotations_v1
├── images
├── LICENSE.txt
├── README.md
├── requirements.txt
├── splits
└── visualize_example.py
"

OUT_DIR="/tinyROI/data/MTSD"

mkdir -p $OUT_DIR

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

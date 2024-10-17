echo "SeaDronesSee"
/SegTrackDetect/scripts/download_SeaDronesSee.sh
python /SegTrackDetect/scripts/converters/SeaDronesSee.py
echo ""

echo "ZebraFish"
/SegTrackDetect/scripts/download_3DZeF20.sh
python /SegTrackDetect/scripts/converters/ZebraFish.py
echo ""

echo "DroneCrowd"
/SegTrackDetect/scripts/download_DroneCrowd.sh
python /SegTrackDetect/scripts/converters/DroneCrowd.py
echo ""

echo "MTSD"
/SegTrackDetect/scripts/download_MTSD.sh
python /SegTrackDetect/scripts/converters/MTSD.py
echo ""
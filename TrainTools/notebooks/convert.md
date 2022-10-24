

cd {ultralytics}/yolov5

python3 gen_wts.py -w best.pt -o test.wts

cd {tensorrtx}/yolov5/

mkdir build

cd build

cp {ultralytics}/yolov5/[test].wts {tensorrtx}/yolov5/build

cmake ..

make

sudo ./yolov5 -s test.wts test.engine m  // serialize model to plan file


// Aimed to convert from [.pt] to [.engine].


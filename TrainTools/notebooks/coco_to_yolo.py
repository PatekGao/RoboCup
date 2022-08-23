import cv2
import json
from pycocotools.coco import COCO


class ConvertCOCOToYOLO:
    """
    Takes in the path to COCO annotations and outputs YOLO annotations in multiple .txt files.
    COCO annotation are to be JSON formart as follows:

        "annotations":{
            "area":2304645,
            "id":1,
            "image_id":10,
            "category_id":4,
            "bbox":[
                0::704
                1:620
                2:1401
                3:1645
            ]
        }
        
    """

    def __init__(self, img_path, json_path):
        self.img_folder = img_path
        self.json_path = json_path
        self.coco = COCO(json_path)

    @staticmethod
    def convert_labels(imgInfo, x1, y1, x2, y2):
        """
        Definition: Parses label files to extract label and bounding box
        coordinates. Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
        """

        def sorting(l1, l2):
            if l1 > l2:
                lmax, lmin = l1, l2
                return lmax, lmin
            else:
                lmax, lmin = l2, l1
                return lmax, lmin

        size = (imgInfo['height'], imgInfo['width'])
        xmax, xmin = sorting(x1, x2)
        ymax, ymin = sorting(y1, y2)
        dw = 1. / size[1]
        dh = 1. / size[0]
        x = (xmin + xmax) / 2.0
        y = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def get_img_info(self, imgId):
        return self.coco.loadImgs([imgId])[0]

    def convert(self, annotation_key='annotations', img_id='image_id', cat_id='category_id', bbox='bbox'):
        # Enter directory to read JSON file
        data = json.load(open(self.json_path))

        check_set = set()

        # Retrieve data
        for i in range(len(data[annotation_key])):

            # Get required data
            image_id = f'{data[annotation_key][i][img_id]}'
            category_id = f'{data[annotation_key][i][cat_id]}'
            bbox = data[annotation_key][i]['bbox']

            imgInfo = self.get_img_info(int(image_id))

            # Retrieve image.
            if self.img_folder is None:
                image_path = f'{imgInfo["file_name"]}.jpg'
            else:
                image_path = f'./{self.img_folder}/{imgInfo["file_name"].split(".")[0]}.jpg'

            # Convert the data
            kitti_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            yolo_bbox = self.convert_labels(imgInfo, kitti_bbox[0], kitti_bbox[1], kitti_bbox[2], kitti_bbox[3])

            # Prepare for export

            filename = f'../data/yolo/{imgInfo["file_name"]}.txt'
            content = f"{category_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}"

            # Export 
            if image_id in check_set:
                # Append to existing file as there can be more than one label in each image
                file = open(filename, "a")
                file.write("\n")
                file.write(content)
                file.close()

            elif image_id not in check_set:
                check_set.add(image_id)
                # Write files
                file = open(filename, "w")
                file.write(content)
                file.close()

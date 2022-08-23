import json
import os
import random
from PIL import Image

with open('../data/task.json', 'r') as file:
	jsonFile = json.load(file)
for i in range(len(jsonFile['images'])-1,-1,-1):
    ram1_num=random.randint(1,3)
    if ram1_num==1:
        ram_num=random.randint(1,3)
        imgpath='../data/image/'+jsonFile['images'][i]['file_name']
        img = Image.open(imgpath)
        if ram_num==1:
            img1 = img.rotate(90, expand = 1)
            new_name = jsonFile['images'][i]['file_name'].split('.')[0]+'_Turn90.jpg'
            newpath='../data/image/'+new_name
        elif ram_num==2:
            img1 = img.rotate(180, expand = 1)
            new_name = jsonFile['images'][i]['file_name'].split('.')[0]+'_Turn180.jpg'
            newpath='../data/image/'+new_name
        else:
            img1 = img.rotate(270, expand = 1)
            new_name = jsonFile['images'][i]['file_name'].split('.')[0]+'_Turn270.jpg'
            newpath='../data/image/'+new_name
        new_width = img1.width
        new_height = img1.height 
        if img1.mode=='P' or img1.mode=='RGBA':
            img1=img1.convert('RGB')
        img1.save(newpath)
        jsonFile['images'][i]['width'] = new_width
        jsonFile['images'][i]['height'] = new_height
        jsonFile['images'][i]['file_name'] = new_name
    else:
        jsonFile['images'].pop(i)
for q in range(len(jsonFile['annotations'])-1,-1,-1):
    flag=0
    img_id = jsonFile['annotations'][q]['image_id']
    for k in range(len(jsonFile['images'])):
        if jsonFile['images'][k]['id'] == img_id:
            this_width=jsonFile['images'][k]['width']
            this_height=jsonFile['images'][k]['height']
            flag=1
            name=jsonFile['images'][k]['file_name']
            if name.split('Turn')[1] == '90.jpg':
                if len(jsonFile['annotations'][q]['segmentation'])!=0:
                    for j in range(len(jsonFile['annotations'][q]['segmentation'][0])):
                        if j%2==0:
                            new_x = jsonFile['annotations'][q]['segmentation'][0][j+1]
                            new_y = this_width - jsonFile['annotations'][q]['segmentation'][0][j]
                            jsonFile['annotations'][q]['segmentation'][0][j] = new_x
                            jsonFile['annotations'][q]['segmentation'][0][j+1] = new_y
                new_bbox_x = jsonFile['annotations'][q]['bbox'][1]
                new_bbox_y = this_width - jsonFile['annotations'][q]['bbox'][0] - jsonFile['annotations'][q]['bbox'][2]
                new_bbox_w = jsonFile['annotations'][q]['bbox'][3]
                new_bbox_h = jsonFile['annotations'][q]['bbox'][2]
                jsonFile['annotations'][q]['bbox'][0]= new_bbox_x
                jsonFile['annotations'][q]['bbox'][1]= new_bbox_y
                jsonFile['annotations'][q]['bbox'][2]= new_bbox_w
                jsonFile['annotations'][q]['bbox'][3]= new_bbox_h
            elif name.split('Turn')[1] == '180.jpg':
                if len(jsonFile['annotations'][q]['segmentation'])!=0:
                    for j in range(len(jsonFile['annotations'][q]['segmentation'][0])):
                        if j%2==0:
                            new_x = this_width - jsonFile['annotations'][q]['segmentation'][0][j]
                            new_y = this_height - jsonFile['annotations'][q]['segmentation'][0][j+1]
                            jsonFile['annotations'][q]['segmentation'][0][j] = new_x
                            jsonFile['annotations'][q]['segmentation'][0][j+1] = new_y
                new_bbox_x = this_width - jsonFile['annotations'][q]['bbox'][2] - jsonFile['annotations'][q]['bbox'][0]
                new_bbox_y = this_height - jsonFile['annotations'][q]['bbox'][3] - jsonFile['annotations'][q]['bbox'][1]
                new_bbox_w = jsonFile['annotations'][q]['bbox'][2]
                new_bbox_h = jsonFile['annotations'][q]['bbox'][3]
                jsonFile['annotations'][q]['bbox'][0]= new_bbox_x
                jsonFile['annotations'][q]['bbox'][1]= new_bbox_y
                jsonFile['annotations'][q]['bbox'][2]= new_bbox_w
                jsonFile['annotations'][q]['bbox'][3]= new_bbox_h
            else:
                if len(jsonFile['annotations'][q]['segmentation'])!=0:
                    for j in range(len(jsonFile['annotations'][q]['segmentation'][0])):
                        if j%2==0:
                            new_x = this_height - jsonFile['annotations'][q]['segmentation'][0][j+1]
                            new_y = jsonFile['annotations'][q]['segmentation'][0][j]
                            jsonFile['annotations'][q]['segmentation'][0][j] = new_x
                            jsonFile['annotations'][q]['segmentation'][0][j+1] = new_y
                new_bbox_x = this_height - jsonFile['annotations'][q]['bbox'][3] - jsonFile['annotations'][q]['bbox'][1]
                new_bbox_y = jsonFile['annotations'][q]['bbox'][0]
                new_bbox_w = jsonFile['annotations'][q]['bbox'][3]
                new_bbox_h = jsonFile['annotations'][q]['bbox'][2]
                jsonFile['annotations'][q]['bbox'][0]= new_bbox_x
                jsonFile['annotations'][q]['bbox'][1]= new_bbox_y
                jsonFile['annotations'][q]['bbox'][2]= new_bbox_w
                jsonFile['annotations'][q]['bbox'][3]= new_bbox_h
    if flag==0:
        jsonFile['annotations'].pop(q)
for i in range(len(jsonFile['images'])):
    old_id=jsonFile['images'][i]['id']
    jsonFile['images'][i]['id'] = i+1
    for j in range(len(jsonFile['annotations'])):
        if jsonFile['annotations'][j]['image_id'] == old_id:
            jsonFile['annotations'][j]['image_id'] = jsonFile['images'][i]['id']
for i in range(len(jsonFile['annotations'])):
    jsonFile['annotations'][i]['id'] = i+1
with open("../data/task0.json", 'w') as f:
    f.write(json.dumps(jsonFile))

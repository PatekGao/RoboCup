import json
from PIL import Image

with open('data/task.json', 'r' ) as file:
	nowFile = json.load(file)
for i in range(len(nowFile['images'])):
    if ~nowFile['images'][i]['file_name'].find('Resize'):
        continue
    imgpath='data/image/'+nowFile['images'][i]['file_name']
    img = Image.open(imgpath)
    img1=Image.new('RGB',(img.size[0]*4,img.size[1]*4),(128,128,128))
    img1.paste(img,(0,0,img.size[0],img.size[1]))
    new_name = nowFile['images'][i]['file_name'].split('.')[0]+'_Resize16x.jpg'
    newpath='data/image/'+new_name
    new_width = img1.width
    new_height = img1.height 
    img1.save(newpath)
    nowFile['images'][i]['width'] = new_width
    nowFile['images'][i]['height'] = new_height
    nowFile['images'][i]['file_name'] = new_name
with open("data/task0.json", 'w') as f:
    f.write(json.dumps(nowFile))
import os, cv2, numpy as np, json
from tqdm import *
from PIL import Image

SIZE = 512
MARGIN = 64
np.set_printoptions(threshold=np.inf, linewidth=999999)

original_images_path = r'E:/LIFULL HOMES DATA (HIGH RESOLUTION)/photo-rent-madori-full-00'

with open(r'./annot_json/instances_train.json', mode='r') as f_train:
    train_jpgs = [_['file_name'] for _ in json.load(f_train)['images']]
with open(r'./annot_json/instances_val.json', mode='r') as f_val:
    val_jpgs = [_['file_name'] for _ in json.load(f_val)['images']]
with open(r'./annot_json/instances_test.json', mode='r') as f_test:
    test_jpgs = [_['file_name'] for _ in json.load(f_test)['images']]
jpgs = {'train': train_jpgs, 'val': val_jpgs, 'test': test_jpgs}

for fpath, dirnames, fnames in tqdm(os.walk(original_images_path)):
    if len(fpath) == len(original_images_path + r'/00/3b/4e9f86428b96821fad75394385ec') and\
        (not ('8486f08035ba152d5244ac54099c' in fpath or 'c468a57377ff8ef63d3b26a6d1fa' in fpath)):
        for i in range(len(fnames)):
            if os.path.exists(
                    os.path.join(r'./annot_npy',
                                 fpath[-34:-32] + '-' + fpath[-31:-29] + '-' + fpath[-28:] + '-' + fnames[i].replace('.jpg', '.npy'))
            ):
                img_original = Image.open(os.path.join(fpath, fnames[i]))

                boundary_path = os.path.join(r'./original_vector_boundary',
                             fpath[-34:-32] + '-' + fpath[-31:-29] + '-' + fpath[-28:] + '-' + fnames[i].replace('.jpg', '.npy'))
                boundary = np.load(boundary_path, allow_pickle=True).item()
                x_min = boundary['x_min']
                x_max = boundary['x_max']
                y_min = boundary['y_min']
                y_max = boundary['y_max']
                width = x_max - x_min
                mid_width = (x_max + x_min) / 2
                height = y_max - y_min
                mid_height = (y_max + y_min) / 2
                if width > height:
                    scale = (SIZE - 2 * MARGIN) / width
                else:
                    scale = (SIZE - 2 * MARGIN) / height
                # print(x_min, y_min, x_max, y_max, width, height, scale)

                original_width, original_height = img_original.size
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                scaled_image = img_original.resize((new_width, new_height), Image.ANTIALIAS)
                canvas = Image.new("RGB", (512, 512), (255, 255, 255))
                # print(new_width, new_height)
                x_topleft_offset = int(512/2 - mid_width * scale)
                y_topleft_offset = int(512/2 - mid_height * scale)
                canvas.paste(scaled_image, (x_topleft_offset, y_topleft_offset))

                for mode in ['train', 'val', 'test']:
                    if fpath[-34:-32] + '-' + fpath[-31:-29] + '-' + fpath[-28:] + '-' + fnames[i] in jpgs[mode]:
                        canvas.save(os.path.join('./' + mode,
                                         fpath[-34:-32] + '-' + fpath[-31:-29] + '-' + fpath[-28:] + '-' + fnames[i].replace('.jpg', '.png')))
                        i2 = cv2.imread(os.path.join('./' + mode,
                                                     fpath[-34:-32] + '-' + fpath[-31:-29] + '-' + fpath[-28:] + '-' + fnames[i].replace('.jpg',
                                                                                                                                         '.png')))
                        os.rename(os.path.join('./' + mode,
                                               fpath[-34:-32] + '-' + fpath[-31:-29] + '-' + fpath[-28:] + '-' + fnames[i].replace('.jpg', '.png')),
                                  os.path.join('./' + mode,
                                               fpath[-34:-32] + '-' + fpath[-31:-29] + '-' + fpath[-28:] + '-' + fnames[i])
                                  )

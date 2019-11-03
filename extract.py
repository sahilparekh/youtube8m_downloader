import pandas as pd
import cv2
import os
from subprocess import check_call
import subprocess
from multiprocessing import Pool, cpu_count
import logging
import random
import imutils


baseurl = 'http://youtu.be/'
temp_dir = 'temp'
output_dir = 'output'
output_dir2 = 'output2'
logging.basicConfig(filename='youtube_scarper.log', level=logging.DEBUG)

def extract_video(item):
    grp, lst = item
    op_vid = grp + '_temp.mp4'
    down_vid = os.path.join(temp_dir, op_vid)
    full_url = baseurl + grp
    # Use youtube_dl to download the video
    try:
        # logging.debug(f"Downloading {grp} video at {down_vid}")
        FNULL = open(os.devnull, 'w')
        check_call(['youtube-dl', \
                    # '--no-progress', \
                    '-f', 'best[ext=mp4]', \
                    '-o', down_vid, \
                    full_url], \
                   stdout=FNULL, stderr=subprocess.STDOUT)
        logging.debug('{0} -- Completed {1} video at {2}'.format(grp, grp, down_vid))
    except subprocess.CalledProcessError:
        logging.error('{0} -- Could not download {1} video'.format(grp, grp))
        return

    # logging.debug(f"Opening {down_vid}")
    vidcap = cv2.VideoCapture(down_vid)
    wanted_idx = lst
    #we choose any two frames from the video to take snapshots of
    if len(lst) > 2:
        wanted_idx = random.choices(lst, k=2)

    count = 1
    for idx in wanted_idx:
        logging.debug('{0} -- Selecting {1}'.format(grp, idx))
        rec = train.loc[idx]
        # logging.debug(f"Fetching Rec {idx}")
        image_name = str(idx) + '_' + rec.cname + '.jpg'
        # logging.debug(f"Created Image name {image_name}")
        opdir = output_dir if count <= 1 else output_dir2
        op_path = os.path.join(opdir, rec.cname)
        if not os.path.isdir(op_path):
            logging.debug('{0} -- Creating dir {1}'.format(grp, op_path))
            os.makedirs(op_path)

        # logging.debug(f"Setting video pos to {str(rec.ts)}ms")
        vidcap.set(cv2.CAP_PROP_POS_MSEC, rec.ts)
        ret, img = vidcap.read()
        if ret:
            #if you would like to resize
            #img = imutils.resize(img, width=350)
            img_path = os.path.join(op_path, image_name)
            logging.debug('{0} -- Frame successfully returned and writing to {1}'.format(grp, img_path))
            cv2.imwrite(img_path, img)
            count = count + 1
        else:
            logging.error('{0} -- Could not write {1}'.format(grp, image_name))

    # logging.debug(f"Releasing vid cap")
    vidcap.release()
    os.remove(down_vid)

train = pd.read_csv('yt_bb_classification_train.csv', header=None)
train.columns=['yt_id', 'ts', 'cid', 'cname', 'present']
all_present = train[train.present == 'present']
train_grouped = all_present.groupby('yt_id').groups
class_list = list(set(train['cname']))
vids_list = list(set(train['yt_id']))

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if not os.path.isdir(temp_dir):
    os.makedirs(temp_dir)

train_iterator = list(train_grouped.items())

# for train_rec in train_iterator:
#     extract_video(train_rec)
threads = cpu_count() - 1
logging.debug('Total Threads {0}'.format(threads))
with Pool(processes=threads) as p:
    p.map(extract_video, train_iterator)






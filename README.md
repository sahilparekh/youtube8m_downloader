# youtube8m_downloader
Youtube-8m Image recognition dataset downloader and tfrecord creator.
The program works in two parts
1. extract.py: scraps the youtube 8m videos and downloads two frames per video (can be changed in program)
2. yt8m_tfrecord.py: creates tfrecord and pbtxt of top n classes which has max images. (can be modified to choose specific classes)

All the vieos are downloaded in temp dir, frames are extracted and then video is deleted. It will also create **pbtxt** file for further use.
**multiprocessing** is used in python to scrap multiple videos at sametime.

## Getting Started

1. Clone this Repo .
2. Download yt8m frame level item csv from:
https://drive.google.com/drive/folders/1_8jXla9b_9SEU6zLCDQADvn2qmL_HNic?usp=sharing
3. Place both the files in the cloned dir
4. Make relevent changes in extract.py and run (preferably in tmux or screen if on server, as it takes very long)
5. Run yt8m_tfrecord.py script:
```
USAGE: yt8m_tfrecord.py [flags]
flags:

yt8m_tfrecord.py:
  --output_dir: Directory path for tfrecords yt8m dataset. will have train and validation subdirectories inside it.
    (default: 'output_tf')
  --pbtxt: path to write pbtxt filepath to write pbtxt file
    (default: 'label.pbtxt')
  --raw_data_dir: Directory path for raw yt8m dataset. Should classes subdirectories inside it.
    (default: 'output')
  --top_n: top n number of classesdefault is 15
    (default: '15')
    (an integer)
  --val_ratio: validation ratio to split datasetdefault is 0.2
    (default: '0.2')
    (a number)
```

### Prerequisites

1. Python 3
2. youtube-dl
3. OpenCV
4. pandas
5. tensorflow
6. imutils (if you would like to resize frames)

##### 1. Python 3 Installation
This you would already know

##### 2. youtube-dl Installation
You will need youtube-dl. Installation instruction can be found on this link [youtube-dl](https://github.com/ytdl-org/youtube-dl)
Still for your quick reference will list installation instruction:

```
sudo -H pip3 install --upgrade youtube-dl
```

##### 3. OpenCV Installation
There are various way to install OpenCV but example using (Conda, PIP or build from source). But for purpose of this project below is instruction using PIP

```
pip3 install opencv-contrib-python
```

##### 4. pandas Installation
```
pip3 install pandas
```

##### 5. tensorflow Installation
```
pip3 install tensorflow
```
or 
```
pip3 install tensorflow-gpu
```

##### 4. imutils Installation
```
pip3 install imutils
```

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details

### Star the REPO, if you find it useful. Feel free for pull requests.
## CHEERS!!! 



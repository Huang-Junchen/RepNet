'''
Author: Huang-Junchen huangjc_mail@163.com
Date: 2023-06-02 12:55:22
LastEditors: Huang-Junchen huangjc_mail@163.com
LastEditTime: 2023-06-02 13:26:14
FilePath: /RepNet/tools/DatasetDownloader.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from pytube import YouTube
import pandas as pd
import os
import shutil
from tqdm import tqdm

DATASET_PATH = 'data/'

CSV_LIST = {'train': 'countix_train.csv', 
            'test' : 'countix_test.csv',
            'val'  : 'countix_val.csv'
            }

def readCSV(csv_filename):
    df = pd.read_csv(DATASET_PATH+csv_filename)
    df = df.iloc[:10]
    return df


def download(video_folder, video_ids):
# 循环遍历视频ID列表
    for index, row in tqdm(video_ids.iterrows(), total=len(video_ids)):
        video_id = row[0]
        video_path = os.path.join(DATASET_PATH, video_folder, video_id+'.mp4')
        if os.path.exists(video_path):
            print(video_id+" has exists.")
            continue   
        try:
            # 创建YouTube对象
            print("Downloading " + video_id)
            yt = YouTube("https://www.youtube.com/watch?v=" + video_id)

            # 获取视频的最高质量
            stream = yt.streams.get_highest_resolution()

            filename = video_id + '.mp4'
            # 下载视频
            stream.download(output_path=os.path.join(DATASET_PATH, video_folder), filename=filename)

            print("Video downloaded successfully")
        except:
            print("Error downloading video")

def build(type):
    print("Read "+ type + " CSV")
    src_folder = os.path.join(DATASET_PATH, ORIGIN_PATH)
    if type == 'train':
        ids = readCSV('train')
        des_folder = os.path.join(DATASET_PATH, TRAIN_PATH)
    elif type == 'test':
        ids = readCSV('test')
        des_folder = os.path.join(DATASET_PATH, TEST_PATH)
    elif type == 'val':
        ids = readCSV('val')
        des_folder = os.path.join(DATASET_PATH, VAL_PATH)
    else:
        print("Unknow type")
        return
    
    if not os.path.exists(des_folder):
        os.makedirs(des_folder)
    
    for index, row in ids.iterrows():
        video_id = row[0]
        src_file = os.path.join(src_folder, video_id+'.mp4')
        if os.path.exists(src_file):
            filename = type+str(index)+'.mp4'
            des_file = os.path.join(des_folder, filename)
            print("Copy "+ video_id + " to " + des_file)
            shutil.copy(src_file, des_file)



if __name__ == '__main__':
    for key, value in CSV_LIST.items():
        df = readCSV(value)
        video_ids = df["video_id"]
        download(key, video_ids)



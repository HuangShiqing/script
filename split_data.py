from sklearn.model_selection import train_test_split
import os
import xml.etree.ElementTree as ET
import sys
from tqdm import tqdm


# <class 'list'>: ['2007_000027.jpg', [486, 500, [['person', 174, 101, 349, 351]]]]
def read_xml(ANN, pick):
    print('Parsing for {}'.format(pick))

    chunks = list()
    no_use = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    # annotations = glob.glob(str(annotations) + '*.xml')
    size = len(annotations)

    # dumps = list()
    # cur_dir = os.getcwd()
    # os.chdir(ANN)
    # path = '/home/hsq/DeepLearning/data/car/bdd100k/daytime.txt'
    # annotations = []
    # with open(path) as fh:
    #     for line in tqdm(fh):
    #         temp = '/home/hsq/DeepLearning/data/car/bdd100k/labels/100k/train_xml/' + line.strip()[-21:].rstrip(
    #             '.jpg') + '.xml'
    #         annotations.append(temp)
    # size = len(annotations)

    for file in tqdm(annotations):

        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        jpg = str(root.find('filename').text) + '.jpg'
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        all = list()

        for obj in root.iter('object'):
            # current = list()
            current = dict()
            name = obj.find('name').text
            if name not in pick:
                continue

            xmlbox = obj.find('bndbox')
            xn = int(float(xmlbox.find('xmin').text))
            xx = int(float(xmlbox.find('xmax').text))
            yn = int(float(xmlbox.find('ymin').text))
            yx = int(float(xmlbox.find('ymax').text))
            # current = [name, xn, yn, xx, yx]
            current['name'] = name
            current['xmin'] = xn
            current['xmax'] = xx
            current['ymin'] = yn
            current['ymax'] = yx
            all += [current]

        add = [[jpg, [w, h, all]]]
        if len(all) is not 0:  # skip the image which not include any 'pick'
            chunks += add
        else:
            no_use.append(add[0][0])
        in_file.close()

    # gather all stats
    stat = dict()
    for dump in chunks:
        all = dump[1][2]
        for current in all:
            if current['name'] in pick:
                if current['name'] in stat:
                    stat[current['name']] += 1
                else:
                    stat[current['name']] = 1

    print('\nStatistics:')
    for i in stat: print('{}: {}'.format(i, stat[i]))
    print('Dataset size: {}'.format(len(chunks)))

    os.chdir(cur_dir)
    return chunks, no_use


def change_label(dumps, src, dst):
    for dump in dumps:
        for i in range(len(dump[1][2])):
            if dump[1][2][i]['name'] == src:
                dump[1][2][i]['name'] = dst

    return dumps

# 按照比率分出训练集和测试集，并且保证每个box都按照比率出现在训练集和测试集合上
from shutil import copy
import numpy as np

ANN = 'D:/DeepLearning/data/WoodBlockNewPick/split/train_xmls/'
picks = ['live knot', 'die knot', 'small', 'head shed', 'edge shed']  # ,
rate = 0.8
chunks, no_use_list = read_xml(ANN, picks)
# np.random.seed(0)
np.random.shuffle(chunks)

os.chdir(ANN)
os.chdir('..')
train_file = open('train_data.txt', 'w')
test_file = open('test_data.txt', 'w')
no_use_file = open('no_use_data.txt', 'w')
# chunks ['744.jpg', [2048, 1536, [{'ymin': 838, 'xmax': 1396, 'ymax': 1061, 'name': 'die knot', 'xmin': 397}]]]
# 把每个box的jpg文件名，按照box的类别分别存到所属的列表里
split_jpg = [list(), list(), list(), list(), list()]
# no_use_jpg = []
for chunk in chunks:
    for i in range(len(chunk[1][2])):
        if chunk[1][2][i]['name'] in picks:
            split_jpg[picks.index(chunk[1][2][i]['name'])].append(chunk[0])

train_list = split_jpg[0][0:int(rate * len(split_jpg[0]))]
for i in range(len(picks) - 1):
    train_list = train_list + split_jpg[i + 1][0:int(rate * len(split_jpg[i + 1]))]
    train_list = list(set(train_list))

test_list = split_jpg[0][int(rate * len(split_jpg[0])):]
for i in range(len(picks) - 1):
    test_list = test_list + split_jpg[i + 1][int(rate * len(split_jpg[i + 1])):]
    test_list = list(set(test_list))

intersection = list(set(train_list).intersection(set(test_list)))
for i in intersection:
    train_list.remove(i)

for i in train_list:
    train_file.write(ANN + i.rstrip('jpg') + 'xml' + '\n')
for i in test_list:
    test_file.write(ANN + i.rstrip('jpg') + 'xml' + '\n')
for i in no_use_list:
    no_use_file.write(ANN + i.rstrip('jpg') + 'xml' + '\n')
train_file.close()
test_file.close()
no_use_file.close()
exit()

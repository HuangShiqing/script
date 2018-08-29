from sklearn.model_selection import train_test_split
import os
import xml.etree.ElementTree as ET
import sys


# <class 'list'>: ['2007_000027.jpg', [486, 500, [['person', 174, 101, 349, 351]]]]
def pascal_voc_clean_xml(ANN, pick, exclusive=False):
    print('Parsing for {} {}'.format(
        pick, 'exclusively' * int(exclusive)))

    dumps = list()
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

    for i, file in enumerate(annotations):
        # progress bar
        sys.stdout.write('\r')
        percentage = 1. * (i + 1) / size
        progress = int(percentage * 20)
        bar_arg = [progress * '=', ' ' * (19 - progress), percentage * 100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()

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
            dumps += add
        in_file.close()

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current['name'] in pick:
                if current['name'] in stat:
                    stat[current['name']] += 1
                else:
                    stat[current['name']] = 1

    print('\nStatistics:')
    for i in stat: print('{}: {}'.format(i, stat[i]))
    print('Dataset size: {}'.format(len(dumps)))

    os.chdir(cur_dir)
    return dumps


def change_label(dumps, src, dst):
    for dump in dumps:
        for i in range(len(dump[1][2])):
            if dump[1][2][i]['name'] == src:
                dump[1][2][i]['name'] = dst

    return dumps


from shutil import copy
import numpy as np

# cwd = 'D:/DeepLearning/data/WoodBlockNewPick/'
# # ANN = 'D:/DeepLearning/data/WoodBlockNewPick/labels/'
# xmls = os.listdir(cwd + 'labels')
#
# # if os.path.isdir(ANN) is False:
# #     os.mkdir(ANN + 'split')
# np.random.shuffle(xmls)
# n = len(xmls)
# rate = 0.8
# train_xmls = xmls[0:int(rate * n)]
# test_xmls = xmls[int(rate * n):]
#
# for i in range(len(xmls)):
#     if (xmls[i] not in train_xmls) and (xmls[i] not in test_xmls):
#         copy(cwd + 'images/' + xmls[i].rstrip('.xml') + '.jpg',
#              cwd + 'split/leftover/' + xmls[i].rstrip('.xml') + '.jpg')
# for i in range(len(train_xmls)):
#     copy(cwd + 'labels/' + train_xmls[i], cwd + 'split/train_xmls/' + train_xmls[i])
#     copy(cwd + 'images/' + train_xmls[i].rstrip('.xml') + '.jpg',
#          cwd + 'split/train_images/' + train_xmls[i].rstrip('.xml') + '.jpg')
# for i in range(len(test_xmls)):
#     copy(cwd + 'labels/' + test_xmls[i], cwd + 'split/test_xmls/' + test_xmls[i])
#     copy(cwd + 'images/' + test_xmls[i].rstrip('.xml') + '.jpg',
#          cwd + 'split/test_images/' + test_xmls[i].rstrip('.xml') + '.jpg')

ANN = '/home/hsq/DeepLearning/data/split/train_xmls/'
picks = ['live knot', 'die knot', 'small', 'head shed']  # 'edge shed',
rate = 0.8
chunks = pascal_voc_clean_xml(ANN, picks, exclusive=False)
np.random.seed(0)
np.random.shuffle(chunks)
# n = len(chunks)

# train_file = open(ANN.rstrip('labels/') + 'train_data.txt', 'w')
# test_file = open(ANN.rstrip('labels/') + 'test_data.txt', 'w')

# ['744.jpg', [2048, 1536, [{'ymin': 838, 'xmax': 1396, 'ymax': 1061, 'name': 'die knot', 'xmin': 397}]]]
class_num = [list(), list(), list(), list()]
for chunk in chunks:
    for i in range(len(chunk[1][2])):
        if chunk[1][2][i]['name'] in picks:
            class_num[picks.index(chunk[1][2][i]['name'])].append(chunk[0])

train_list = class_num[0][0:int(0.8 * len(class_num[0]))]
for i in range(len(picks)-1):
    train_list = train_list + class_num[i+1][0:int(0.8 * len(class_num[i+1]))]
    train_list = list(set(train_list))

test_list = class_num[0][int(0.8 * len(class_num[0])):]
for i in range(len(picks)-1):
    test_list = test_list + class_num[i+1][int(0.8 * len(class_num[i+1])):]
    test_list = list(set(test_list))

test = train_list + test_list
test = list(set(test))
exit()
# for dump in chunks[0:int(0.8 * n)]:
#     train_file.write(ANN + dump[0].rstrip('jpg') + 'xml' + '\n')
# for dump in chunks[int(0.8 * n):]:
#     test_file.write(ANN + dump[0].rstrip('jpg') + 'xml' + '\n')
# train_file.close()
# test_file.close()

# dumps = change_label(dumps, 'live knot', 'knot')
# dumps = change_label(dumps, 'die knot', 'knot')

# ['2007_000027.jpg', [486, 500, [['person', 174, 101, 349, 351]]]]
# image_path = dumps[:]
# x_train, x_valid, y_train, y_valid = train_test_split(ful_image_path, ful_labels,
#                                                               test_size=(valid_proportion + test_proportion),
#                                                                stratify=ful_labels, random_state=1)


exit()

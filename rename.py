# coding:utf-8
import os
import argparse


def sort_in_int(l):
    # 从数组中删除名字不带‘jpg’的文件
    for i in range(len(l) - 1, -1, -1):  # 必须得倒序删除元素，这样才不会数组越界
        if 'jpg' not in l[i]:
            del l[i]

    for i in range(len(l)):
        l[i] = l[i].split('.')
        l[i][0] = int(l[i][0])
    l.sort()
    for i in range(len(l)):
        l[i][0] = str(l[i][0])
        l[i] = l[i][0] + '.' + l[i][1]
    return l


if __name__ == '__main__':
    # parser.add_argument(
    #     '--zero', type=str, help='if there is 0 in the name', default='no'
    # )
    # FLAGS = parser.parse_args()

    old_names = os.listdir('./', )
    old_names = sort_in_int(old_names)
    count_i = 0

    for old_name in old_names:
        if "jpg" not in old_name:
            continue

        new_name = str(count_i) + ".jpg"
        os.rename('./' + old_name, './' + new_name)
        count_i += 1

    # flage_repeat = True
    # while flage_repeat is True:
    #     flage_repeat = False
    #     old_names = os.listdir('./train/',)#('D:/DeepLearning/data/WoodBlockNew')  #
    #     count_i = 0
    #
    #     for old_name in old_names:
    #         if "jpg" not in old_name:
    #             continue
    #
    #         new_name = str(count_i) + ".jpg"
    #         try:
    #             os.rename('./train/' + old_name, './train/' + new_name)
    #         except FileExistsError:
    #             new_name = str(count_i) + '.(1)' + ".jpg"
    #             os.rename('./train/' + old_name, './train/' + new_name)
    #             flage_repeat = True
    #         count_i += 1

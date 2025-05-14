"""
data_preprocess.py
数据预处理脚本
使用BaseDT库将flower_dataset数据集转化为IMAGENET格式，且train与val的比例为8:2

注1：BaseDT要求预先将图片分类好，把图片放入images文件夹，并建立classes.txt文件。
注2：直接使用0.8:0:0.2的比例会导致test数据集各个类别被分配了一张图片（可能是浮点数错误？），且无法生成train.txt文件，解决方法1为使用0:0.8:0.2的比例，再把test改成train，解决方法2为抛弃BaseDT，从头造轮子。
"""

from BaseDT.dataset import DataSet
ds = DataSet(r"Ex1/data") # 指定为生成数据集的路径
ds.make_dataset(r"flower_dataset", src_format="IMAGENET",train_ratio = 0.8, test_ratio = 0 , val_ratio=0.2)# 指定原始数据集的路径，数据集格式选择IMAGENET

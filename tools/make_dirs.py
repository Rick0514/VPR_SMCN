import os 
import sys

# ---------------------------- 说明 ----------------------------------
# 自动生成文件目录
# 论文用到了四个数据集和两个描述子
# 如果目录下没有别的目录，其对应的值应设为None
# 用法：python make_dirs.py <你的路径>
# ---------------------------- 说明 ----------------------------------


datasets = [
    'nordland',
    'gardens_point',
    'ox_robotcar',
    'scut'
]

net = [
    'netvlad',
    'alexnet'
]

layer = {}
for each_net in net:
    layer[each_net] = datasets

dirs = {
    'model': None,
    'result': layer,
    'desc': layer,
    'datasets': datasets
}


# make dirs

def make_dir_recursively(path, value):
    if value:
        if type(value) == list:
            for each in value:
                ddir = os.path.join(path, each)
                if not os.path.exists(ddir):
                    os.mkdir(ddir)
        elif type(value) == dict:
            for each in value.keys():
                ddir = os.path.join(path, each)
                if not os.path.exists(ddir):
                    os.mkdir(ddir)
                make_dir_recursively(ddir, value[each])
        else:
            return
    
    return


if __name__ == '__main__':
    make_dir_recursively(sys.argv[0], dirs)





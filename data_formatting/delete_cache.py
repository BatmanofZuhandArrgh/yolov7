import os
import glob as glob

if __name__ == '__main__':
    paths = glob.glob('/home/iasrl/Documents/**/*.cache', recursive=True)
    for path in paths:
        os.remove(path)
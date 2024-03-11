import os
import sys

class init_env:
    def __init__(self):
        pass
    def __call__(self):
        os.environ['PYFLEXROOT']="/home/luhr/correspondence/softgym++"+'/Pyflex'
        os.environ['PYTHONPATH']=os.environ['PYFLEXROOT']+'/bindings/build'
        os.environ['LD_LIBRARY_PATH']=os.environ['PYFLEXROOT']+'/external/SDL2-2.0.10/lib/x64'
        sys.path.append(os.getcwd())

if __name__ == '__main__':
    init_env()()
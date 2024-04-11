#!/bin/bash

# 1. install Baseline
cd Baseline && 
rm -rf build &&
python setup.py install &&
cd ../ &&

# 2. install Baseline_kernel
cd Baseline_kernel && 
rm -rf build &&
python setup.py install &&
cd ../ &&

# 3. compile Magicsphere kernel
cd Magicsphere/build1 &&
# cmake
cmake -DCMAKE_INSTALL_PREFIX=../ .. &&
make &&
make install &&
rm -rf ./* &&

# 4. insatll Magicsphere pytorch 
cd ../ &&
rm -rf build &&
python setup.py install
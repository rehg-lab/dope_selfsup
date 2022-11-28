#!/bin/bash                                   

echo Downloading data...

wget https://dl.dropbox.com/s/9ku4whbhk4iqzvk/ABC_dope-selfsup.tar
wget https://dl.dropbox.com/s/m3jmhhnfbo9cftn/ModelNet_dope-selfsup.tar
wget https://dl.dropbox.com/s/7dylsrndq3srq1z/splits.tar

echo Extracting data...
tar -xf ABC_dope-selfsup.tar
tar -xf ModelNet_dope-selfsup.tar
tar -xf splits.tar

mkdir dataset_directory
cd dataset_directory

ln -s ../ABC_dope-selfsup abc
ln -s ../ModelNet_dope-selfsup modelnet

cd ..

echo Making lowshot split with symlinks...
python prep/make_symlinked_dataset.py --src_p=dataset_directory/modelnet --dest_p=dataset_directory/modelnet_lowshot

cd dataset_directory/modelnet_lowshot

echo Extracting splits... 
tar -xf ../../splits.tar

cd ../..

rm ABC_dope-selfsup.tar
rm ModelNet_dope-selfsup.tar
rm splits.tar

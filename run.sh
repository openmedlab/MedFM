export WORKDIR=/medfmc_exp
cd $WORKDIR
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/train.py configs/densenet/dense121_chest.py --work-dir work_dirs/temp/

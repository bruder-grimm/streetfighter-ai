pip3 install -r ./requirements.txt --upgrade
python3 -m retro.import roms
python3 play.py --model model_weights_0.85_1.0_0.01_best_test.hdf5
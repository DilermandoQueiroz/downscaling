python prepare_dataset.py --time_end 2004
wait
python prepare_dataset.py --time_init 2004 --time_end 2007 --type val
wait
python prepare_dataset.py --time_init 2007 --time_end 2014 --type test


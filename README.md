# baseline-mmoe

#baseline_PEPLER:
from utils.py to load data and use data_index to split data into train/val/test datasets
module.py is for PEPLER model pipeline
run reg.py


#MMOE:
from data_prepare to generate train/val/test index for baseline data alignment(save index for next-time load) and load data.
group_tr_mmoe is the train/val/test file
diff_attr_mmoe.py is for my MMOE model pipeline
run run_mmoe.py for train/val/test

#peter_data.py is for transfer our data index to original baseline data(reviews.pickle) for alignment

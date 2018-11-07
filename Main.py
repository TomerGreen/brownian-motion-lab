from data_loader import *
import pandas



def test(path):
    f = pandas.read_csv(path)
    print(f.head())


dirpath = "D:\GDrive\Lab II\WEEK 2"
path = "D:\GDrive\Lab II\Raw Data/1.csv"

# save_data_from_dir(dirpath)
test(path)

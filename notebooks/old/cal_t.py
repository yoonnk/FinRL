import pandas as pd

data_df = pd.read_excel(r'C:\Users\USER\Desktop\FinRL\data\DataforRL.xlsx',
                        usecols=['pressure', 'flowrate', 'Total'], nrows=3876)
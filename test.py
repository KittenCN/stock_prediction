import numpy as np
from datetime import datetime, timedelta

def generate_dates(start_date, num_days):
    start_date = datetime.strptime(start_date, '%Y%m%d') # Convert string to datetime
    if num_days < 0:
        return np.array([ (start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(num_days, 1) ])
    else:
        return np.array([ (start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(num_days+1) ])

# Test the function
dates = generate_dates('20230509', -10)
print(dates)

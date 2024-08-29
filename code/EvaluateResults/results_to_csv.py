import os
import pandas as pd

print(os.getcwd())
base_file_path = '../data/results/winoqueer/'

# list all files in directory starting with "summary"
files = [f for f in os.listdir(base_file_path) if f.startswith('summary')]
# create two separate dataframes for files with gender and sexuality in the name
files = [f for f in files if 'noeval' not in f]

categories = ['gender', 'sexual']

for cat in categories:
    cat_files = [f for f in files if cat in f]
    results = []
    header = []
    for file in cat_files:
        with open (base_file_path + file, 'r') as f:
            model_name = file.split('_')[1].split(cat)[0]
            mode = file.split('-')[-1].split('.')[0]
            if mode == "prompt":
                mode = "soft-prompt"
            result = f.readlines()[-2:]
            terms = result[0].split(' ')[-1].strip().split(',')
            scores = [float(x) for x in result[1].strip().split(',')]
            if not header:
                header = ['model', 'mode'] + terms
            results.append([model_name, mode] + scores)
    df = pd.DataFrame(results, columns=header)
    df.to_csv(f'../data/results/winoqueer/all_summary_{cat}.csv', index=False)




def secs_to_hrs_mins(x):
    x = float(x)
    hours = x // 3600
    minutes = (x % 3600) // 60
    if minutes < 10:
        minutes = f'0{int(minutes)}'
    else:
        minutes = f'{int(minutes)}'
    return f'{int(hours)}:{minutes}'


def hrs_mins_to_secs(x):
    x = x.split(':')
    return int(x[0]) * 3600 + int(x[1]) * 60



def time_to_co2(x, a100):
    secs = hrs_mins_to_secs(x)
    hrs = secs / 3600
    carb_eff = 0.432
    return round(((250 * hrs * carb_eff)/1000) * a100,2)

times = ['25:09', '32:04', '24:13', '24:08', '32:47', '22:52']

total_co2 = 0
for time in times:
    co2 = time_to_co2(time, 4)
    print(f'{time} {co2}')
    total_co2 += co2
print(total_co2)




#
results = []

for cat in categories:
    cat_files = [f for f in files if cat in f]
    for file in cat_files:
        with open (base_file_path + file, 'r') as f:
            model_name = file.split('_')[1].split(cat)[0]
            mode = file.split('-')[-1].split('.')[0]
            if mode == "prompt":
                mode = "soft-prompt"
            result = f.readlines()[2]
            time_sec = result.split(',')[-1].strip()
            time = secs_to_hrs_mins(time_sec)
            co2 = round(time_to_co2(time, 2), 2)
            print(f'{model_name} {mode} {cat} {time} {co2}')
            results.append([model_name, mode, cat, time, co2])
            print()

header = ['model', 'mode', 'category', 'time', 'co2']
df = pd.DataFrame(results, columns=header)
sum_Co2 = df['co2'].sum()

# parse time column to get total time
df['time'] = df['time'].apply(lambda x: x.split(':')).apply(lambda x: int(x[0]) * 3600 + int(x[1]) * 60)

sum_time = df['time'].sum()
sum_time = secs_to_hrs_mins(sum_time)
print(f'Total time: {sum_time}')
print(f'Total CO2: {sum_Co2} kg')
df.to_csv(f'../data/results/winoqueer/all_summary_time_co2.csv', index=False)


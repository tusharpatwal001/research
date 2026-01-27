# %%
# insertion sort

a = [2, 4, 4, 1, 2, -3, -7]


# %%
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        for j in range(i, 0, -1):
            if arr[j - 1] > arr[j]:
                print("j -> ", j)
                arr[j - 1], arr[j] = arr[j], arr[j - 1]
            else:
                break
            break
    return arr


# %%
insertion_sort(a)

# %%
# prime finder using list comprehension
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97

# False if num == 1 else True if len([i for i in range(2, int(num**.5)+1) if (num % i) == 0]) == 0 else False
# []
# num > 1 and all(num % i for i in range(2, int(num**0.5) + 1))


# %%
num = 13
num > 1 and all(num % i for i in range(2, int(num**0.5) + 1))


# %%

# num = 2
arr = [
    4,
    232,
    12,
    21,
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
]

[
    (
        (i, "prime")
        if i
        in [
            num
            for num in arr
            if num > 1 and all(num % i for i in range(2, int(num**0.5) + 1))
        ]
        else (i, "composite")
    )
    for i in arr
]


# %%
arr = [2, 12, 32, 43, 54, 54, 6, 56, 5, 656, 5]

[
    (
        (i, "P")
        if i
        in [
            num
            for num in arr
            if num > 1 and all(num % i for i in range(2, int(num**0.5) + 1))
        ]
        else (i, "C")
    )
    for i in arr
]


# %%

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

[item for element in matrix for item in element]


# %%

1000**0.5


# %%


def s(arr: list):
    n = len(arr)
    flag = True
    while flag:
        flag = False
        for i in range(1, n):
            if arr[i] < arr[i - 1]:
                flag = True
                arr[i], arr[i - 1] = arr[i - 1], arr[i]

    return arr

arr = [1, 3, 4, 34, 2,32, 23, 23]
# %%
s(arr)

# %%


def selection_sort(arr: list):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                arr[j], arr[min_idx] = arr[min_idx], arr[j]
    return arr


selection_sort(arr)




# %%




# %% 
# import opendatasets as op


# op.download("https://www.kaggle.com/competitions/playground-series-s6e1", data_dir="data")
# timepass000013
# 9b8ebd1ee2f5a10df8288fb0540b32bf443

# %%
import pandas as pd

df = pd.read_csv("./data/playground-series-s6e1/train.csv")
df1 = pd.read_csv("./data/playground-series-s6e1/test.csv")


print(df.head())


# %%

# OneHotEncoder
# nominal data 
df['gender'].unique()
# ['female', 'other', 'male']

# %%

# OneHotEncoder
# nominal data 
df['course'].unique()
# ['b.sc', 'diploma', 'bca', 'b.com', 'ba', 'bba', 'b.tech']
# %%

# OneHotEncoder
# nominal data
df['internet_access'].unique()
# ['no', 'yes']


# %%
# OneHotEncoder
# nominal data 
df['study_method'].unique()
# ['online videos', 'self-study', 'coaching', 'group study', 'mixed']




# %%

# OrdinalEncoder
# ordinal data
df['sleep_quality'].unique()
# ['average', 'poor', 'good']

# %%

# OrdinalEncoder
# ordinal data
df['facility_rating'].unique()
# ['low', 'medium', 'high']

# %%

# OrdinalEncoder
# ordinal data
df['exam_difficulty'].unique()
# ['easy', 'moderate', 'hard']

# %%

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

ordinal_encoded = ordinal_encoder.fit_transform(df1.iloc[:, [8, 10, 11]])

ordinal_df = pd.DataFrame(
    ordinal_encoded,
    columns=ordinal_encoder.get_feature_names_out(df1.columns[[8, 10, 11]])
)
ordinal_df.shape
# %%

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

encoded = ohe.fit_transform(df1.iloc[:, [2, 3, 6, 9]])

encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(df1.columns[[2, 3, 6, 9]])
)
encoded_df.shape
# %%

input_data = pd.concat([df.iloc[:, [1, 4, 5, 7]], ordinal_df, encoded_df], axis=1)
# output_data 
output_data = df1.iloc[:, -1]

# %%
input_data_test = pd.concat([df1.iloc[:, [1, 4, 5, 7]], ordinal_df, encoded_df], axis=1)


input_data_test.shape


# %%
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(input_data, output_data)
# %%

pred_output = reg.predict(input_data_test)
# %%


pd.DataFrame({"id": df1['id'], "exam_score": pred_output}).to_csv("data/playground-series-s6e1/output1.csv", index=False)

# %%
# df1['id'].to_list()
pred_output
# %%

# %%
from uu import encode
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import opendatasets as op


op.download(
    "https://www.kaggle.com/competitions/playground-series-s6e1", data_dir="data")
# timepass00001
# 9b8ebd1ee2f5a10df8288fb0540b32bf

# %%
import pandas as pd
df = pd.read_csv("./data/playground-series-s6e1/train.csv")
df1 = pd.read_csv("./data/playground-series-s6e1/test.csv")


print(df.head())


# %%
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

def encode_data(data):

    ordinal_encoder = OrdinalEncoder()
    ordinal_encoded = ordinal_encoder.fit_transform(data.iloc[:, [8, 10, 11]])
    ordinal_df = pd.DataFrame(
        ordinal_encoded, columns=ordinal_encoder.get_feature_names_out(data.columns[[8, 10, 11]]))

    ohe = OneHotEncoder(sparse_output=False)
    encoded = ohe.fit_transform(data.iloc[:, [2, 3, 6, 9]])
    encoded_df = pd.DataFrame(
        encoded, columns=ohe.get_feature_names_out(data.columns[[2, 3, 6, 9]]))

    input_data = pd.concat(
        [data.iloc[:, [1, 4, 5, 7]], ordinal_df, encoded_df], axis=1)
    return input_data


# %%

# training dataset
x_train = encode_data(df)
y_train = df.iloc[:, -1]

# test dataset
x_test = encode_data(df1)



# output_data
print(x_train.shape, y_train.shape, x_test.shape)

# %%

# %%
# from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.svm import SVR

reg = SVR().fit(x_train, y_train)
y_pred = reg.predict(x_train.iloc[:100, :])
y_true = y_train.iloc[:100]

print("rmse -", root_mean_squared_error(y_true, y_pred))

 # %%
 
 
y_pred = reg.predict(x_train.iloc[:2000, :])
y_true = y_train.iloc[:2000]

print("rmse -", root_mean_squared_error(y_true, y_pred))
 # %%

pred_output = reg.predict(x_test)
pd.DataFrame({"id": df1['id'], "exam_score": pred_output}).to_csv(
    "data/playground-series-s6e1/output4.csv", index=False)

# %%

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

def load_compensation_data():
    file = Path("Employee_Compensation_20250829.csv")
    return pd.read_csv(file, low_memory=False)

def print_model(model):
    return list(zip(model.feature_names_in_.tolist(), model.coef_.tolist()[0])), model.intercept_[0]

def plot_data(df, independent, dependent):
    df.plot(kind="scatter", x=independent, y=dependent, grid=True, alpha=0.2)
    plt.show()

def print_error(true_values, predicted_values):
    test_mse = mean_squared_error(y_true=true_values, y_pred=predicted_values)
    test_rmse = root_mean_squared_error(y_true=true_values, y_pred=predicted_values)
    print("Mean squared error: " + str(test_mse) + " root mean squared error: " + str(test_rmse))

def main():
    compensation = load_compensation_data()
    compensation.dropna(inplace=True)
    #remove commas and convert to numeric
    compensation["Total Benefits"] = pd.to_numeric(compensation["Total Benefits"].str.replace(",", ""), errors="coerce")
    compensation["Total Salary"] = pd.to_numeric(compensation["Total Salary"].str.replace(",", ""), errors="coerce")

    #plot data
    plot_data(compensation, "Total Salary", "Total Benefits")

    #create train and test data
    train_set, test_set = train_test_split(compensation, test_size = 0.2)

    label = "Total Benefits"
    train_label = train_set[[label]]
    features = ["Total Salary"]
    train_features = train_set[features]
    model = LinearRegression()
    model.fit(train_features, train_label)
    test_label = test_set[[label]]
    test_features = test_set[features]

    #create prediction
    test_pred = model.predict(test_features)

    #create new plot
    test_salary = test_features["Total Salary"].to_numpy()
    test_benefits = test_label["Total Benefits"].to_numpy()
    test_pred_values = test_pred.flatten()
    test_df = pd.DataFrame({'salary': test_salary, 'benefits': test_benefits, 'pred': test_pred_values})
    fig, ax = plt.subplots()
    test_df.plot(kind='scatter', x = ['salary'], y=['benefits'], grid=True, alpha=0.2, ax=ax, color='b')
    test_df.plot(kind='scatter', x=['salary'], y=['pred'], grid=True, alpha=0.2, ax=ax, color='g')
    plt.show()

    print_error(test_label, test_pred)

    joblib.dump(value=model, filename="model.pkl")

main()
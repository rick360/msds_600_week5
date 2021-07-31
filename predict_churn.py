def load_data(filepath):
    """
    Loads churn data into dataframe from string filepath.
    """
    df = pd.read_csv('prepped_churn_data.csv', index_col ='customerID')
    return df

def make_predictions(df):
    """
    Uses pycaret best model to make predictions on data in the df dataframe
    """
    model = load_model('ABC')
    predictions = predict_model(model, data=df)
    predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'}, inplace=True)
    return predictions['Churn_prediction']

if __name__ == "__main__":
    df= load_data('new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
    
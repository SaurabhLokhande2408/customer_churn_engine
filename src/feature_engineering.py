import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def feature_eng(train_df: pd.DataFrame, test_df: pd.DataFrame = None):
    train_df = train_df.copy()

    columns_to_label_encode = [
        'Gender', 'Married', 'Internet Service', 'Online Security',
        'Device Protection Plan', 'Paperless Billing', 'Under 30',
        'Senior Citizen', 'Multiple Lines', 'Premium Tech Support',
        'Streaming TV', 'Streaming Movies', 'Streaming Music',
        'Unlimited Data', 'Churn Label', 'Online Backup'
    ]

    columns_to_drop_1 = ['Dependents', 'Total Revenue', 'Customer Status', 'Under 30']
    columns_to_drop_2 = ['Customer ID', 'Zip Code', 'Latitude', 'Longitude', 'Referred a Friend']
    columns_to_drop_3 = ['Churn Category', 'Churn Reason', 'Churn Score', 'City','Satisfaction Score']

    columns_to_one_hot_encode = [
        'Country', 'State', 'Quarter', 'Offer', 'Phone Service',
        'Internet Type', 'Contract', 'Payment Method'
    ]

    if test_df is not None:
        test_df = test_df.copy()

        for col in columns_to_label_encode:
            if col in train_df.columns:
                le = LabelEncoder()
                train_df[col] = le.fit_transform(train_df[col])
                test_df[col] = le.transform(test_df[col])

        train_df.drop(columns=columns_to_drop_1, inplace=True)
        train_df.drop(columns=columns_to_drop_2, inplace=True)
        train_df.drop(columns=columns_to_drop_3, inplace=True)

        test_df.drop(columns=columns_to_drop_1, inplace=True)
        test_df.drop(columns=columns_to_drop_2, inplace=True)
        test_df.drop(columns=columns_to_drop_3, inplace=True)

        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        train_encoded = pd.DataFrame(
            ohe.fit_transform(train_df[columns_to_one_hot_encode]),
            columns=ohe.get_feature_names_out(columns_to_one_hot_encode),
            index=train_df.index
        )
        test_encoded = pd.DataFrame(
            ohe.transform(test_df[columns_to_one_hot_encode]),
            columns=ohe.get_feature_names_out(columns_to_one_hot_encode),
            index=test_df.index
        )

        train_df = pd.concat([train_df.drop(columns=columns_to_one_hot_encode), train_encoded], axis=1)
        test_df = pd.concat([test_df.drop(columns=columns_to_one_hot_encode), test_encoded], axis=1)

        return train_df, test_df

    # Fallback for single dataframe
    for col in columns_to_label_encode:
        if col in train_df.columns:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])

    train_df.drop(columns=columns_to_drop_1, inplace=True)
    train_df.drop(columns=columns_to_drop_2, inplace=True)
    train_df.drop(columns=columns_to_drop_3, inplace=True)

    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    train_encoded = pd.DataFrame(
        ohe.fit_transform(train_df[columns_to_one_hot_encode]),
        columns=ohe.get_feature_names_out(columns_to_one_hot_encode),
        index=train_df.index
    )
    train_df = pd.concat([train_df.drop(columns=columns_to_one_hot_encode), train_encoded], axis=1)

    return train_df

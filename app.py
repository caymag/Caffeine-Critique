from flask import Flask, render_template, request
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from gpytorch.kernels import RBFKernel
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam

app = Flask(__name__)

# function to extract metadata from markdown files
def extract_metadata_from_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    pattern = re.compile(r'---\n(.*?)\n---', re.DOTALL)
    match = pattern.search(content)
    if match:
        metadata = match.group(1)
        metadata_dict = {}
        for line in metadata.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata_dict[key.strip()] = value.strip().strip('"')
        return metadata_dict
    else:
        return {}

@app.route('/', methods=["GET", "POST"])
def home():
    prediction = None 
    if request.method == "POST":
        # Get form data
        price = request.form.get("price")
        roast_level = request.form.get("roast_level")
        espresso = request.form.get("espresso")
        sweetness = request.form.get("sweetness")
        strength = request.form.get("strength")
        house_syrups = request.form.get("house_syrups")
        specialty_drinks = request.form.get("specialty_drinks")
        espresso_variety = request.form.get("espresso_variety")
        mixed = request.form.get("mixed")
        edible_decor = request.form.get("edible_decor")

        # Directory containing markdown files
        directory = "Coffee Shops"

        # List to store extracted data
        data = []

        # Loop through files in the directory
        for file_name in os.listdir(directory):
            if file_name.endswith('.md'):
                file_path = os.path.join(directory, file_name)
                metadata = extract_metadata_from_markdown(file_path)
                if metadata:
                    metadata['file name'] = file_name[:len(file_name) - 3]
                    data.append(metadata)

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)

        columns_to_keep = ['file name', 'location', 'rating', 'price', 'roast level', 'espresso', 'sweetness', 'strength', 'house syrups', 'specialty drinks', 'espresso variety', 'mixed', 'edible decor']

        # Select only the specified columns that exist
        df = df[[col for col in columns_to_keep if col in df.columns]]

        # Encoding categorical variables
        label_encoder = LabelEncoder()
        label_encoder.fit(['Light', 'Medium', 'Medium-dark', 'Dark'])
        df['roast level'] = label_encoder.transform(df['roast level'])
        
        ordinal_encoder_price = OrdinalEncoder(categories=[['Expensive', 'Average', 'Cheap']])
        df['price'] = ordinal_encoder_price.fit_transform(df[['price']])

        ordinal_encoder_espresso = OrdinalEncoder(categories=[['Horrible', 'Mediocre', 'Good', 'Exceptional']])
        df['espresso'] = ordinal_encoder_espresso.fit_transform(df[['espresso']])

        ordinal_encoder_sweetness = OrdinalEncoder(categories=[['Subtle', 'Sweet', 'Balanced']])
        df['sweetness'] = ordinal_encoder_sweetness.fit_transform(df[['sweetness']])

        ordinal_encoder_strength = OrdinalEncoder(categories=[['Weak', 'Strong', 'Balanced']])
        df['strength'] = ordinal_encoder_strength.fit_transform(df[['strength']])

        one_hot_encoder = OneHotEncoder()
        encoded_columns = one_hot_encoder.fit_transform(df[['house syrups', 'specialty drinks', 'espresso variety', 'mixed', 'edible decor']])
        encoded_df = pd.DataFrame(encoded_columns.toarray(), columns=one_hot_encoder.get_feature_names_out(['house syrups', 'specialty drinks', 'espresso variety', 'mixed', 'edible decor']))
        df = pd.concat([df, encoded_df], axis=1)

        df.drop(columns=['house syrups', 'specialty drinks', 'espresso variety', 'mixed', 'edible decor'], inplace=True)

        # Prepare the data for training
        x = df.drop(columns=['file name', 'rating', 'location'])
        y = df['rating']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=63)

        # Botorch mixed model uses a specialized kernel for categorical data 
        categorical_columns = ['roast level', 'price', 'espresso', 'sweetness', 'strength']
        categorical_columns += list(one_hot_encoder.get_feature_names_out(['house syrups', 'specialty drinks', 'espresso variety', 'mixed', 'edible decor']))

        cat_dims = [x.columns.get_loc(col) for col in categorical_columns if col in x.columns]

        # Standardize the target data (rating)
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

        # Convert arrays to tensors; use 64 bit floating point number
        x_train_tensor = torch.tensor(x_train.values, dtype=torch.float64)
        x_test_tensor = torch.tensor(x_test.values, dtype=torch.float64)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float64).squeeze()
        y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float64).squeeze()

        # Create the MixedSingleTaskGP model
        class MyMixedSingleTaskGP(MixedSingleTaskGP):
            def __init__(self, x_train_tensor, y_train_tensor, cat_dims):
                cont_kernel = RBFKernel
                
                def cont_kernel_factory(**kwargs):
                    return cont_kernel
                
                # Specify likelihood function
                likelihood = GaussianLikelihood()
                
                super().__init__(x_train_tensor, y_train_tensor, cat_dims=cat_dims, 
                                cont_kernel_factory=cont_kernel_factory,
                                likelihood=likelihood)

        model = MyMixedSingleTaskGP(x_train_tensor, y_train_tensor.unsqueeze(-1), cat_dims=cat_dims)

        # Optimize
        optimizer = Adam(model.parameters(), lr=.01)  
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        NUM_EPOCHS = 150

        model.train()
        for epoch in range(NUM_EPOCHS):
            optimizer.zero_grad() 
            output = model(x_train_tensor) 
            loss = -mll(output, y_train_tensor) 
            loss.backward()  
            optimizer.step() 

        model.eval()

        # Prepare the new data point
        new_data = {
            'price': price,
            'roast level': roast_level,
            'espresso': espresso,
            'sweetness': sweetness,
            'strength': strength,
            'house syrups': house_syrups,
            'specialty drinks': specialty_drinks,
            'espresso variety': espresso_variety,
            'mixed': mixed,
            'edible decor': edible_decor
        }

        # Preprocess the new data point
        new_data_df = pd.DataFrame([new_data])
        new_data_df['roast level'] = label_encoder.transform(new_data_df['roast level'])

        new_data_df['price'] = ordinal_encoder_price.transform(new_data_df[['price']])
        new_data_df['espresso'] = ordinal_encoder_espresso.transform(new_data_df[['espresso']])
        new_data_df['sweetness'] = ordinal_encoder_sweetness.transform(new_data_df[['sweetness']])
        new_data_df['strength'] = ordinal_encoder_strength.transform(new_data_df[['strength']])

        new_encoded_columns = one_hot_encoder.transform(new_data_df[['house syrups', 'specialty drinks', 'espresso variety', 'mixed', 'edible decor']])
        new_encoded_df = pd.DataFrame(new_encoded_columns.toarray(), columns=one_hot_encoder.get_feature_names_out(['house syrups', 'specialty drinks', 'espresso variety', 'mixed', 'edible decor']))

        new_data_df = pd.concat([new_data_df.drop(columns=['house syrups', 'specialty drinks', 'espresso variety', 'mixed', 'edible decor']), new_encoded_df], axis=1)

        new_x_tensor = torch.tensor(new_data_df.values, dtype=torch.float64)

        # Make prediction
        with torch.no_grad():
            posterior = model.posterior(new_x_tensor)
            pred_mean = posterior.mean.item()
            pred_std = posterior.variance.sqrt().item()

        # Inverse transform the prediction to original scale
        pred_mean_orig = target_scaler.inverse_transform(np.array([[pred_mean]]))
        prediction = {
            'mean': pred_mean_orig[0][0],
            'std': pred_std
        }

    return render_template("form.html", prediction=prediction)

if __name__ == '__main__':
    app.run()
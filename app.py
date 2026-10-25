from flask import Flask, render_template, request, abort
import pandas as pd
import os
import re
import numpy as np
import torch
from gpytorch.kernels import RBFKernel
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam
from waitress import serve

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

columns_to_keep = ['file name', 'location', 'rating', 'price', 'roast level', 'espresso', 'sweetness', 'strength', 'house syrups', 'specialty drinks', 'espresso variety', 'mixed', 'edible decor', 'bitter']

# Select only the specified columns that exist
df = df[[col for col in columns_to_keep if col in df.columns]]

shops = df[['file name', 'location', 'rating']].assign(rating=pd.to_numeric(df['rating']))
cities = sorted(shops['location'].unique())

# Yes/No fields, one-hot encoded
binary_columns = ['house syrups', 'specialty drinks', 'espresso variety', 'mixed', 'edible decor', 'bitter']

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
encoded_columns = one_hot_encoder.fit_transform(df[binary_columns])
encoded_df = pd.DataFrame(encoded_columns.toarray(), columns=one_hot_encoder.get_feature_names_out(binary_columns))
df = pd.concat([df, encoded_df], axis=1)

df.drop(columns=binary_columns, inplace=True)

# Prepare the data for training
x = df.drop(columns=['file name', 'rating', 'location'])
y = df['rating']

# Botorch mixed model uses a specialized kernel for categorical data 
categorical_columns = ['roast level', 'price', 'espresso', 'sweetness', 'strength']
categorical_columns += list(one_hot_encoder.get_feature_names_out(binary_columns))

cat_dims = [x.columns.get_loc(col) for col in categorical_columns if col in x.columns]

# Standardize the target data (rating)
target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1))

# Convert arrays to tensors; use 64 bit floating point number
x_train_tensor = torch.tensor(x.values, dtype=torch.float64)
y_train_tensor = torch.tensor(y_scaled, dtype=torch.float64).squeeze()

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

# Form values the fitted encoders accept; anything else would crash transform()
ALLOWED_VALUES = {
    'roast level': set(label_encoder.classes_),
    'price': set(ordinal_encoder_price.categories_[0]),
    'espresso': set(ordinal_encoder_espresso.categories_[0]),
    'sweetness': set(ordinal_encoder_sweetness.categories_[0]),
    'strength': set(ordinal_encoder_strength.categories_[0]),
}
for col, cats in zip(one_hot_encoder.feature_names_in_, one_hot_encoder.categories_):
    ALLOWED_VALUES[col] = set(cats)

@app.route('/', methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        new_data = {
            'price': request.form.get('price'),
            'roast level': request.form.get('roast_level'),
            'espresso': request.form.get('espresso'),
            'sweetness': request.form.get('sweetness'),
            'strength': request.form.get('strength'),
            'house syrups': request.form.get('house_syrups'),
            'specialty drinks': request.form.get('specialty_drinks'),
            'espresso variety': request.form.get('espresso_variety'),
            'mixed': request.form.get('mixed'),
            'edible decor': request.form.get('edible_decor'),
            'bitter': request.form.get('bitter')
        }
        for key, value in new_data.items():
            if value not in ALLOWED_VALUES[key]:
                abort(400, f"Invalid value for {key}")

        # Preprocess the new data point
        new_data_df = pd.DataFrame([new_data])
        new_data_df['roast level'] = label_encoder.transform(new_data_df['roast level'])

        new_data_df['price'] = ordinal_encoder_price.transform(new_data_df[['price']])
        new_data_df['espresso'] = ordinal_encoder_espresso.transform(new_data_df[['espresso']])
        new_data_df['sweetness'] = ordinal_encoder_sweetness.transform(new_data_df[['sweetness']])
        new_data_df['strength'] = ordinal_encoder_strength.transform(new_data_df[['strength']])

        new_encoded_columns = one_hot_encoder.transform(new_data_df[binary_columns])
        new_encoded_df = pd.DataFrame(new_encoded_columns.toarray(), columns=one_hot_encoder.get_feature_names_out(binary_columns))

        new_data_df = pd.concat([new_data_df.drop(columns=binary_columns), new_encoded_df], axis=1)

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
            'std': pred_std * target_scaler.scale_[0]
        }

    return render_template("form.html", prediction=prediction)

@app.route('/shops')
def shop_list():
    city = request.args.get('city', '')
    sort = request.args.get('sort', 'desc')
    rows = shops[shops['location'] == city] if city else shops
    rows = rows.sort_values('rating', ascending=(sort == 'asc'))
    return render_template("shops.html", shops=rows.to_dict('records'), cities=cities, city=city, sort=sort)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
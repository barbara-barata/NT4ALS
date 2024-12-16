import os
import json
import pandas as pd
import numpy as np

# Path to the folder containing JSON files
folder_path = "/Users/bsbar/OneDrive/Ambiente de Trabalho/docs tese/MIRA/RAW DATA"
output_excel = 'RAW_DATA.xlsx'

# Initialize an empty list to hold the rows
rows = []
wavelengths = []

# Loop through each JSON file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)

        # Open and read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extract the required data
        sample_origin_params = data.get("Sample Origin Parameters", {})
        ab_data = data.get("AB Data", [])
        data_status_params = data.get("Data Status Parameters", {})

        # Calculate wavelengths only once
        if not wavelengths:
            fxv = data_status_params.get("FXV")
            lxv = data_status_params.get("LXV")
            npt = data_status_params.get("NPT")

            wavelengths = [
                fxv - ((i * (fxv - lxv)) / (npt - 1))
                for i in range(npt)
            ]

        # Create a row with 'NAM', 'INS', and AB Data
        row = [sample_origin_params.get("NAM"), sample_origin_params.get("INS")] + ab_data
        rows.append(row)

# Convert the rows to a DataFrame
df = pd.DataFrame(rows)

# Add column names
columns = ["NAM", "INS"] + wavelengths
df.columns = columns

# Save the DataFrame to an Excel file
df.to_excel(output_excel, index=False)

print(f"Master spreadsheet created with wavelengths: {output_excel}")

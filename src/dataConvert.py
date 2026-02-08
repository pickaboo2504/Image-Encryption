import csv

def transpose_csv(input_file, output_file):
    # Read the CSV file
    with open(input_file, 'r', newline='') as f:
        reader = list(csv.reader(f))  # Convert to list for indexing
    
    # Ensure there is data
    if not reader:
        print("Error: CSV file is empty.")
        return
    
    # Transpose the data
    transposed_data = list(map(list, zip(*reader)))  

    # Write transposed data to new CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(transposed_data)

# Example usage
input_csv = 'input.csv'   # Replace with your input file path
output_csv = 'output.csv' # Replace with your desired output file path
transpose_csv(input_csv, output_csv)

print("CSV file transposed successfully!")

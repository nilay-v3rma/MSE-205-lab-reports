import csv

# File paths
input_file = 'Compression_CastIron.txt'  # Replace with the full path to your .TRA file
output_file = 'MSE205_Stainless Steel.csv'

# Read the .TRA file
with open(input_file, 'r') as file:
    lines = file.readlines()

# Locate the header line and data
header_line_index = 5  # Adjust based on the structure of the file
header = lines[header_line_index].strip().split(';')
data_lines = lines[header_line_index + 1:]

# Write to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header)  # Write the header
    for line in data_lines:
        csv_writer.writerow(line.strip().split(';'))

print(f"Data has been successfully converted to {output_file}")

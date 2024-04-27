import csv

def remove_parentheses(input_csv_file, output_csv_file):
    with open(input_csv_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        with open(output_csv_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            for row in reader:
                modified_row = [field.replace('(', '').replace(')', '') for field in row]
                writer.writerow(modified_row)

input_file = 'eigenvectors.csv'
output_file = 'eigenvectors_mat.csv'

remove_parentheses(input_file, output_file)
print("Parentheses removed from the CSV file and saved to", output_file)

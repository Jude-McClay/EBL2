import csv
import os
import fnmatch

def remove_parentheses(input_csv_file, output_csv_file):
    with open(input_csv_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        with open(output_csv_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            for row in reader:
                modified_row = [field.replace('(', '').replace(')', '') for field in row]
                writer.writerow(modified_row)

#input_file = 'eigenvectors.csv'
#output_file = 'eigenvectors_mat.csv'

#remove_parentheses(input_file, output_file)
#print("Parentheses removed from the CSV file and saved to", output_file)

def process_files(input_directory):
    for filename in os.listdir(input_directory):
        if fnmatch.fnmatch(filename, '*eigenvectors.csv'):
            out_filename = "{0}_{2}.{1}".format(*filename.rsplit('.', 1) + ["mat"])
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(input_directory, out_filename)
            remove_parentheses(input_file, output_file)
            print(filename, out_filename)

def main():
    # process input dir
    process_files('/home/jude/Documents/PhD/CDT/EBL2/python_files/haldane_hamiltonians_2_results')
    print("Processing completed.")

if __name__ == "__main__":
    main()

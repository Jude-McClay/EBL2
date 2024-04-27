import os
import numpy as np
import csv

def read_hermitian_matrix_from_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        matrix = []
        for row in reader:
            matrix.append([complex(val.replace("i", "j")) for val in row])
        return np.array(matrix)

def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def process_matrices(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_directory, filename)
            print(input_file)
            output_prefix = os.path.splitext(filename)[0]

            # Read the Hermitian matrix from CSV
            matrix = read_hermitian_matrix_from_csv(input_file)

            # Compute eigenvectors and eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)

            # Write eigenvalues to CSV without brackets
            eigenvalues = [[val] for val in eigenvalues]
            eigenvalues_output_file = os.path.join(output_directory, f"{output_prefix}_eigenvalues.csv")
            write_to_csv(eigenvalues, eigenvalues_output_file)

            # Convert 'j' to 'i' in eigenvectors and write each eigenvector as a separate column
            eigenvectors = [[str(val).replace('j','i') for val in vec] for vec in eigenvectors]
            eigenvectors_output_file = os.path.join(output_directory, f"{output_prefix}_eigenvectors.csv")
            write_to_csv(eigenvectors, eigenvectors_output_file)

def main():
    # specify the input and output directories
    input_directory = '/home/jude/Documents/PhD/CDT/EBL2/python_files/haldane_hamiltonians_2'
    output_directory = input_directory + "_results"
    print(output_directory)
    process_matrices(input_directory, output_directory)
    print("Processing completed. Results saved in:", output_directory)

if __name__ == "__main__":
    main()

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
        for row in data:
            writer.writerow(row)

def main():
    # Read the Hermitian matrix from CSV
    matrix = read_hermitian_matrix_from_csv("H.csv")

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Convert 'j' to 'i' in eigenvectors
    eigenvectors = [[str(val).replace('j', 'i') for val in vec] for vec in eigenvectors]

    # Write eigenvalues to CSV
    write_to_csv([[val] for val in eigenvalues], "eigenvalues.csv")

    # Write eigenvectors to CSV
    write_to_csv(eigenvectors, "eigenvectors.csv")

if __name__ == "__main__":
    main()

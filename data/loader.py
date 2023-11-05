import glob
import csv


def load_dataset():
    train = []
    file_paths = glob.glob("data/train/*.csv")
    for file_path in file_paths:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip the header
            for row in reader:
                train.append(row[0])

    valid = []
    file_paths = glob.glob("data/valid/*.csv")
    for file_path in file_paths:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip the header
            for row in reader:
                valid.append(row[0])

    return {"train": train, "valid": valid}

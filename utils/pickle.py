import pickle


def dump_pkl_file(obj, path):
    """Serialize and save object to a file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Dumped file to: {path}")


def load_pkl_file(path):
    """Load and deserialize object from a file."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"Loaded file from: {path}")
    return obj

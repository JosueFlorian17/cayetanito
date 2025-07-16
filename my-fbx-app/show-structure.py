import os

def print_structure(root_dir, indent=0):
    for item in sorted(os.listdir(root_dir)):
        if item == "node_modules":
            continue
        path = os.path.join(root_dir, item)
        print("  " * indent + "|-- " + item)
        if os.path.isdir(path):
            print_structure(path, indent + 1)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print_structure(base_dir)
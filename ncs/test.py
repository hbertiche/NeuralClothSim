import os
if __name__ == "__main__":
    print(__file__)
    
    print(os.path.dirname(__file__))
    print(os.path.join(os.path.dirname(__file__), ".."))
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(ROOT_DIR)
    CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
    print(CHECKPOINTS_DIR)
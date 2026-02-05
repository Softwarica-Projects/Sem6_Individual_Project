import sys
import os


def main():
    os.system('python train/train_model.py')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)

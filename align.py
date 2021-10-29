import v4_align_ch4_single
import os
from tqdm import tqdm
import argparse

# wrap logic in function with paths as inputs
def align(input_dir: str, output_dir: str):
    files = []
    for file in os.listdir(input_dir):
        if file.endswith(".nc"):
            files.append(file)

    for i in tqdm(range(len(files))):
        try:
            v4_align_ch4_single.Alignment(os.path.join(input_dir,files[i]),output_dir)
        # catch any 'Exception'
        except Exception as e:
            # Use f-string to print exception and message
            # f-strings are new as of python 3.6+ (relatively recently) 
            # and are a new fancy way to format strings
            # https://realpython.com/python-f-strings/
            print(f'Error ({e}) on file {os.path.join(output_dir,files[i])}')

if __name__ == "__main__":
    # argparse is a python standard library for parsing command line input args
    parser = argparse.ArgumentParser(description="Align")
    parser.add_argument("--input-dir", required=True, help="Input directory with netcdf files")
    parser.add_argument("--output-dir", required=True, help="Where to write output files")
    args = parser.parse_args()
    
    # call align func with the user specified paths
    align(args.input_dir, args.output_dir) 

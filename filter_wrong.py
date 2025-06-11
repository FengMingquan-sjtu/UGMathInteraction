

# read the json file, filter the data dicts with "correctness"==false, and write the filtered data to a new json file
import json
import os
import argparse
from utils import mkdir
def filter_wrong(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    filtered_data = [item for item in data if item.get("correctness")== False]

    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Filtered data written to {output_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', 
                        type=str,
                        help="The input JSON file to filter.",
                        required=True)
    parser.add_argument('--output_file',
                        type=str,
                        help="The output JSON file to write filtered data.",
                        required=True)
    
    args = parser.parse_args()

    mkdir(os.path.dirname(args.output_file))
    filter_wrong(args.input_file, args.output_file)

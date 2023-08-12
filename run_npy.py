import argparse
import os

if __name__ == "__main__":
    
    # Add a command line parser
    parser = argparse.ArgumentParser(description='Run the llama2 using just numpy')
    required_args = parser.add_argument_group('Required arguments')
    required_args.add_argument("-w", '--weight', type=str, help='Path to bin file containing the weights', required=True)
    args = parser.parse_args()

    # Get the weight filepath from parser
    weight_filepath = args.weight

    # Check if the path is valid
    
    # Check that file is a bin file
    if not weight_filepath.endswith('.bin'):
        raise ValueError('The weight file must be a bin file')

    # Check if the file exists
    if not os.path.exists(weight_filepath):
        raise ValueError('The weight file does not exist')


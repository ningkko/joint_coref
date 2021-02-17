import sys
import logging
import argparse

parser = argparse.ArgumentParser(description='summarize scorer output')

parser.add_argument('--input_path', type=str,
                    help='The path to the scorer output file')

args = parser.parse_args()

global placed_holder_printed 
placed_holder_printed = 0

def print_place_holder():
	global placed_holder_printed
	if placed_holder_printed == 1:
		return
	else:
		print("\n...\n")
		placed_holder_printed = 1

with open(args.input_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
    	if "Entity" not in line:
    		print(line.strip("\n"))
    		placed_holder_printed = 0
    	else:
    		print_place_holder()







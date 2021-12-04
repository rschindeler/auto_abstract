# Auto Abstract
Tool for generating abstract block representations of an image.
This is done by:
- Generating a series of candidate "splits", where the area is divided into polygons
- For each candidate split, picking the average RGB value of each sub-ploygon
- Picking the candidate split that results in the lowest error with respect to the original image
- Repeating the process recursively until the end conditions are met

## Installation
### Dependencies
Requires Python3 and Pipenv.
> pip install pipenv
For list of Python package dependencies, see Pipfile.
### Install Auto Abstract 
```
git checkout ...
pipenv install
```

## Usage
The following commands assume the pipenv shell has been activated.
> pipenv shell
To see program help, run:
> python auto_abstract.py --help

### Basic Usage
- Use the --image argument to specify the input image
- Use the --show argument to display the result, or --out to specify the output image file
- Use the --out_width and --out_height to control the shape of the output image

### Modifying Behaviour
The program has a few arguments that can modify the behaviour of the algorithm.

*Recursion Limit Arguments*
Splits that would result in a polygon under these thresholds are discarded.
- --athresh: minimum area of a polygon, 
- --rthresh: minimum ration of width/height of a polygon 
The maximum number of iterations can also be specified by the --iterations argument.

*Split Arguments*
These argument modify how a polygon is split into sub-polygons.
- --tdepth: only consider splitting into triangles after this depth
- --rec_n: Number of ways to split a rectangle
- --tri_n: Number of ways to split a triangle

*Other Argument*
- --max_pixels: Downsample input image to reduce computation cost

## Examples
*Controlling Polygon Size*
> python auto_abstract.py --image examples/input/lake.png --out examples/output/lake.png --athresh 0.01
*Using Triangles*
This works "best" when triangles are used on the last or second last iteration
> python auto_abstract.py --image examples/input/mountain.png --athresh 0.005 --iterations 5 --tdepth 4 --out examples/output/mountain.png
*Multiple Splits*
> python auto_abstract.py --image examples/input/beach.png --rec_n 20 --show --athresh 0.01

"""
Tool for creating abstract representations of images by recursivly splitting ploygons.
Run script with --help for usage.
"""
import os
import sys
import argparse
import math
import yaml

import numpy as np
from PIL import Image
from matplotlib import pyplot
from polygon import Polygon

def main():
    """
    Main method for the auto_abstract tool
    Either load and process the input image, or run a test.
    """
    args = get_args()
    if args.image:
        img = load_image(args.image, max_pixels=args.max_pixels)
        print("Loaded %s with shape: %s" % (args.image, img.shape))
        process_image(img, args)
    elif args.square:
        img = np.zeros((args.square, args.square * 2, 3))
        run_test(img)

def get_args():
    """
    Method to load and validate command-line arguments
    """

    # Load defaults from yaml
    defaults = get_defaults()
    # Example usage string
    example_text='''example:
  python auto_abstract.py --image input_image.jpg --iterations 10 --athresh 0.01 --rthresh 0.01 --show
'''
    # Build argument parser that shows default values and example text
    parser = argparse.ArgumentParser(
        epilog=example_text,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Input arguments
    parser.add_argument('--square', type=int, help='used for debugging')
    parser.add_argument('--image', help='input image')
    # Arguments that control recursvie stop conditions
    parser.add_argument('--iterations', type=int, default=defaults['iterations'], help='maximum number of recursive iterations')
    parser.add_argument('--athresh', type=float, default=defaults['athresh'], help='minimum polygon area')
    parser.add_argument('--rthresh', type=float, default=defaults['rthresh'], help='minimum ratio of width/height')
    # Arguments that affect which polygons are considered
    parser.add_argument('--tdepth', type=int, default=defaults['tdepth'], help='only consider triangles after this iteration')
    parser.add_argument('--rec_n', type=int, default=defaults['rec_n'], help='number of ways to split rectangles')
    parser.add_argument('--tri_n', type=int, default=defaults['tri_n'], help='number of ways to split triangles')
    # Arguments that control the image size
    parser.add_argument('--max_pixels', type=int, default=defaults['max_pixels'], help='if input image has more than this number of pixels, it will be shrunk before processing')
    parser.add_argument('--out_width', type=int, help='width of the output image in pixels')
    parser.add_argument('--out_height', type=int, help='height of the output image in pixels')
    # Output arguments
    parser.add_argument('--out', type=str, help='filename to write output image to')
    parser.add_argument('--show', action='store_true', help='show output image when done')
    args = parser.parse_args()

    if args.image:
        assert os.path.isfile(args.image), 'could not find image %s' % args.image

    if not args.out and not args.show:
        print("Run the script with the --show argument, --out argument, or both")
        sys.exit()

    return args

def process_image(img, args):
    """
    Top-level method to process an input image, given the arguments
    Creates the initial outline Polygon class, and calls the recursive_split() method
    Draws the resulting polygons onto a new canvas and creates output image
    """
    # Create outline Polygon
    total_area = img.shape[0] * img.shape[1]
    outline = Polygon.image_outline(img)

    # Recursive ploygon split based on original image
    print("Processing image with area %s" % outline.area())
    config = {
        'area_threshold': args.athresh,
        'ratio_threshold': args.rthresh,
        'max_iterations': args.iterations,
        'triangle_depth':  args.tdepth,
        'rec_n':  args.rec_n,
        'tri_n':  args.tri_n,
        'rectangle_depth': 0,
        'triangle_method': 'grid',
        'rectangle_method': 'grid',
    }
    polys, colours = outline.recursive_split_polygon(img, config)

    # Create new image by drawing each polygon with the correct colour
    new_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=int)
    black = np.array([0,0,0], dtype=int)
    white = np.array([255,255,255], dtype=int)
    for i in range(len(polys)):
        polys[i].draw(new_img,fill=colours[i],outline=colours[i])

    # Resize the output image
    output_image = prepare_output(new_img, args)
    if args.show:
        display_rgb_image(np.array(output_image))
    if args.out:
        output_image.save(args.out)

def run_test(img):
    """
    Debugging method with sample code for reference after initial return
    """
    outline = Polygon.image_outline(img)
    print(outline.area())
    points = outline.grid_rectangle_points(N=5,img_draw=img)
    display_rgb_image(img)
    return
    choices = outline.grid_triangle_split(N=2)
    Polygon.draw_polygon_split_choices(choices, img)
    display_rgb_image(img)
    return

    return
    total_area = img.shape[0] * img.shape[1]

    polys, colours = outline.split_polygon(img, method='rectangle', random=True)
    outline=np.array([0,0,0])
    for i in range(len(polys)):
        polys[i].draw(img,fill=colours[i],outline=outline)
    display_rgb_image(img)

    return


    #points = outline.points_along_axes(img_draw=img)
    #new_polys = outline.random_triangle_split()
    split_choices = outline.axis_triangle_split(N=10)
    #split_choices = outline.random_rectangle_split(N=100)
    #Polygon.draw_polygon_split_choices(split_choices, img)
    choice = split_choices[5]
    poly = choice[0]
    poly.draw(img, fill=True, outline=True)
    display_rgb_image(img)
    return
    Polygon.draw_polygon_split_choices(split_choices, img)
    return

    center = outline.center_point()
    triangles = outline.new_triangular_polygons(center)
    colours = [
        [255,0,0],
        [0,255,0],
        [0,0,255],
        [255,255,255],
    ]
    for i in range(len(triangles)):
        c, mse = triangles[i].get_overlapping_colour(img)
        triangles[i].draw(img, outline=colours[i], fill=c)
        print("Colour: %s, mean squared error: %s" % (c, mse))
    display_rgb_image(img)
    return

    #outline.draw(img)
    second_polys = []
    threshold = int(total_area * 0.2)
    for p in new_polys:
        p.draw(img)
        #if p.area() > threshold:
        #    second_polys += p.random_triangle_split()
    #for p in second_polys:
        #print(p.v)
    #    p.draw(img)

    display_rgb_image(img)

def load_image(filename, max_pixels=1000000):
    """
    Method to use Pillow to read image and resize
    """
    # Load image
    pil_img = Image.open(filename)

    # Resize if required
    width, height = pil_img.size
    n_pixels = width*height
    if n_pixels > max_pixels:
        ratio = math.sqrt(float(max_pixels) / float(n_pixels)) 
        n_width = int(width*ratio)
        n_height = int(height*ratio)
        print("Resizeing input image from %sx%s to %sx%s (ratio=%s)" % (
            width,
            height,
            n_width,
            n_height,
            ratio))
        pil_img = pil_img.resize((n_width, n_height))

    # Convert to numpy array
    np_img = np.array(pil_img)
    return np_img

def prepare_output(np_img, args):
    """
    Method to convert numpy array back to Pillow and resize
    """
    # Convert from numpy to Pillow
    pil_img = Image.fromarray(np_img.astype(np.uint8))
    width, height = pil_img.size
    if args.out_width:
        resize = True
        # Use width directly
        out_width = args.out_width
        # If both are provided, use explicitly
        if args.out_height:
            out_height = args.out_height
        # If only width is provided, calculate height
        else:
            out_height = int(height * (float(out_width) / float(width)))
    elif args.out_height:
        resize = True
        # Use height directly
        out_height = args.out_height
        # Calculate width 
        out_width = int(width * (float(out_height) / float(height)))
    else:
        out_width = width
        out_height = height
        resize = False

    if resize:
        pil_img = pil_img.resize((out_width, out_height))

    return pil_img

def display_rgb_image(data):
    """
    Displays the image using 'nearest' interpolation - keep pixels sharp
    """
    pyplot.imshow(data, interpolation='nearest')
    pyplot.show()

def get_defaults():
    """
    Load the default command line arguments
    """
    with open('defaults.yaml', 'r') as f:
        defaults = yaml.safe_load(f)
    return defaults



if __name__ == '__main__':
    main()

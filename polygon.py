import copy
import random
import math
import numpy as np
from matplotlib.path import Path

from skimage.draw import line_aa

import multiprocessing

class Polygon:
    def __init__(self, verticies):
        assert type(verticies) == np.ndarray, 'verticies must be a Nx2 array'
        assert len(verticies.shape) == 2, 'verticies must be a Nx2 array'
        assert verticies.shape[1] == 2, 'verticies must be a Nx2 array'

        self.v = copy.deepcopy(verticies)
        self.shape = self.v.shape

    """
    Method to return the center point
    """
    def center_point(self):
        center = np.average(self.v, axis=0)
        return center.astype(int)

    """
    Method to get points inside the polygon along axes drawn from
    each vertex to the center point
    """
    def points_along_axes(self,N=3, img_draw=None):
        points = []
        # Get the center point
        center = self.center_point()
        points.append(center.astype(int))
        # Draw line from each vertex to center to create more poitns
        for i in range(self.shape[0]):
            line = np.array([
                [self.v[i,0], self.v[i,1]],
                [center[0], center[1]]
            ])
            x_len = line[1,0] - line[0,0]
            y_len = line[1,1] - line[0,1]
            x_spacing = x_len / (N+1)
            y_spacing = y_len / (N+1)

            for k in range(N+1):
                point = line[0,:] + [x_spacing*k, y_spacing*k]
                points.append(point.astype(int))

        # Optionally draw on image (debugging)
        if img_draw is not None:
            for p in points:
                img_draw[p[0], p[1], :] = np.array([255,255,255])

        return points

    """
    Method to get random points inside the ploygon
    """
    def random_points(self, N=10, img_draw=None):
        if self.shape[0] == 3:
            points = []
            for _ in range(N):
                # Compute weighted average of the three vertices, with random weights in interval [0,1]
                s, t = sorted([random.random(), random.random()])
                pt0 = self.v[0,:]
                pt1 = self.v[1,:]
                pt2 = self.v[2,:]

                point = np.array(
                        [ s * pt0[0] + (t-s)*pt1[0] + (1-t)*pt2[0],
                          s * pt0[1] + (t-s)*pt1[1] + (1-t)*pt2[1] ]
                )

                points.append(point.astype(int))
        elif self.shape[0] == 4:
            # Split into two triangles and call for each 
            # This is required if the 4-sided polygon is not rectangular
            t1 = Polygon(np.array([self.v[0,:], self.v[1,:], self.v[2,:]]))
            t2 = Polygon(np.array([self.v[2,:], self.v[3,:], self.v[0,:]]))
            n1 = N//2
            n2 = N - n1
            points = t1.random_points(N=n1)
            points += t2.random_points(N=n2)
        else:
            raise RuntimeError("Polygon can only have 3 or 4 verticies")

        # Optionally draw on image (debugging)
        if img_draw is not None:
            for p in points:
                img_draw[p[0], p[1], :] = np.array([255,255,255])

        return points

    """
    Method to split randomly into rectangles
    Args:
        N: (int) Equivilant to N in grid_rectangle_split() for random points inside the polygon 
    """
    def random_rectangle_split(self,N=5):
        if not self.shape[0] == 4:
            raise RuntimeError("rectangle split can only be called on a rectangular polygon")

        # Get a list of points inside the polygon
        points = self.random_points(N=N*N)
        # Generate a series of options for splitting this Polygon up into smaller Polygons
        # using each point as the 'seed'
        choices = []
        for point in points:
            # Create a series of rectangles using this point as a vertex
            choices.append(self.new_rectangular_polygons(point))
        return choices

    """
    Method to split into rectangles along a grid
    Args:
        N: (int) Number splits to consider along each side of the rectangle
    """
    def grid_rectangle_split(self, N=5):
        if not self.shape[0] == 4:
            raise RuntimeError("rectangle split can only be called on a rectangular polygon")

        # Get a list of points along the outline of the polygon
        # points = self.circumference_points(N=N, n_sides=2)

        # Get points in a grid inside the rectangle (this includes the circurmference)
        points = self.grid_rectangle_points(N=N)
        # Generate a series of options for splitting this Polygon up into smaller Polygons
        # using each point as the 'seed'
        choices = []
        for point in points:
            # Create a series of rectangles using this point as a vertex
            choices.append(self.new_rectangular_polygons(point))
        return choices


    """
    Method to randomly split into triangles
    Args:
        N: (int) Equivilant to N in grid_triangle_split() for random points inside the polygon 
    """
    def random_triangle_split(self,N=3):
        # Get a list of points inside the polygon
        points = self.random_points(N=N*N)
        # Generate a series of options for splitting this Polygon up into smaller Polygons
        # using each point as the 'seed'
        choices = []
        for point in points:
            # Create a series of triangle using adjacent two verticies and a point from the list 
            choices.append(self.new_triangular_polygons(point))
        return choices

    """
    Method to randomly split into triangle using a grid
    Args:
        N: (int) Number splits to consider along each side of the triangle
    """
    def grid_triangle_split(self,N=3):
        # Get a list of points inside the polygon
        points = self.circumference_points(N=N)
        # Generate a series of options for splitting this Polygon up into smaller Polygons
        # using each point as the 'seed'
        choices = []
        for point in points:
            # Create a series of triangle using adjacent two verticies and a point from the list 
            choices.append(self.new_triangular_polygons(point))
        return choices

    """
    Method to split into triangles along the major axes of the polygon
    Args:
        N: (int) Number of choices to generate
    """
    def axis_triangle_split(self,N=3):
        # Get a list of points inside the polygon
        points = self.points_along_axes(N=N)
        # Generate a series of options for splitting this Polygon up into smaller Polygons
        # using each point as the 'seed'
        choices = []
        for point in points:
            # Create a series of triangle using adjacent two verticies and a point from the list 
            choices.append(self.new_triangular_polygons(point))
        return choices

    """
    Method to create new rectangular Polygons from each vertix and a specified point
    """
    def new_rectangular_polygons(self, point):
        assert type(point) == np.ndarray, 'point must be a 2-element array'
        assert point.shape == (2,), 'point must be a 2-element array'

        new_polys = []
        # Loop through each vertex
        for i in range(self.shape[0]):
            # Don't add if the vertex has the same x/y as the point
            if self.v[i,0] == point[0] or self.v[i,1] == point[1]:
                continue
            poly = Polygon(np.array([
                [self.v[i,0],self.v[i,1]],
                [self.v[i,0],point[1]],
                [point[0],point[1]],
                [point[0],self.v[i,1]]
            ]))
            new_polys.append(poly)

        return new_polys


    """
    Method to create points in a grid inside a rectangle
    """
    def grid_rectangle_points(self, N=5, img_draw=None):
        # Get the min, max values for x and y
        xmax = self.v[:,0].max(axis=0)
        xmin = self.v[:,0].min(axis=0)
        ymax = self.v[:,1].max(axis=0)
        ymin = self.v[:,1].min(axis=0)

        # Create equally spaced points along x, y
        xpoints = np.linspace(xmin, xmax, N).astype(int)
        ypoints = np.linspace(ymin, ymax, N).astype(int)

        # Use meshgrid to get the actuall [x,y] points
        mx, my = np.meshgrid(xpoints,ypoints)
        # Create a list of individual points 
        points = []
        for i in range(mx.shape[0]):
            for k in range(mx.shape[1]):
                points.append(np.array([mx[i,k], my[i,k]]))

        # Optionally draw on image (debugging)
        if img_draw is not None:
            for p in points:
                img_draw[p[0], p[1], :] = np.array([255,255,255])

        return points



    """
    Method to get points along a the outline
    Args:
        N: (int) Number of points to pick along each line of the polygon
        n_sides: (int) optional limit on how many sides are used to draw points
        img_draw: (numpy array) optional image to draw on
    """
    def circumference_points(self, N=3, n_sides=None, img_draw=None):
        # Determine which sides to consider
        if n_sides is not None:
            assert type(n_sides) == int
            assert n_sides >= 1
            # Use only these sides (loop starts at -1)
            stop_index = n_sides-1
        else:
            # Use all sides (loop starts at -1)
            stop_index = self.shape[0]-1

        points = []
        # Draw line from each vertex to the next (start at -1 to avoid awkward indexing)
        for i in range(-1, stop_index):
            line = np.array([
                [self.v[i,0], self.v[i,1]],
                [self.v[i+1,0], self.v[i+1,1]]
            ])
            x_len = line[1,0] - line[0,0]
            y_len = line[1,1] - line[0,1]
            x_spacing = x_len / (N+1)
            y_spacing = y_len / (N+1)

            for k in range(N+1):
                point = line[0,:] + [x_spacing*k, y_spacing*k]
                points.append(point.astype(int))

        # Optionally draw on image (debugging)
        if img_draw is not None:
            for p in points:
                img_draw[p[0], p[1], :] = np.array([255,255,255])

        return points

    """
    Method to create new triangular Polygons from each vertix and a specified point
    """
    def new_triangular_polygons(self, point):
        assert type(point) == np.ndarray, 'point must be a 1x2 array'
        assert point.shape == (2,), 'point must be a 2-element array'
        new_polys = []
        # Loop through each vertex (start at -1 to avoid awkward indexing at the end of the loop)
        for i in range(-1, self.shape[0]-1):
            poly = Polygon(np.array([
                [self.v[i,0], self.v[i,1]],
                [self.v[i+1,0], self.v[i+1,1]],
                [point[0], point[1]]
            ]))
            if poly.area() == 0:
                continue
            new_polys.append(poly)

        return new_polys

    """
    Method to calculate the area
    """
    def area(self):
        x = self.v[:,0]
        y = self.v[:,1]
        return 0.5 * np.abs(np.dot(x,np.roll(y,1)) - np.dot(y,np.roll(x,1)))

    """
    Method to calculate the streth ratio
    This is the minimum ratio of height/width, width/height
    """
    def stretch_ratio(self):
        w = self.v[:,0].max(axis=0) - self.v[:,0].min(axis=0)
        h = self.v[:,1].max(axis=0) - self.v[:,1].min(axis=0)
        if w == 0 or h == 0:
            return None
        ratio = min([w/h, h/w])
        # print("%sx%s - %s" % (w,h, ratio))
        return ratio

    """
    Draw the polygon onto the image
    """
    def draw(self, img, outline=None, fill=None):
        if type(outline) == bool:
            outline = np.array([255, 255, 255]) * int(outline)
        if type(fill) == bool:
            outline = np.array([255, 255, 255]) * int(fill)

        # Fill in the pixels 
        if fill is not None:
            # Get the mask indicating which pixels are part of the polygon
            #  - the mask is just drawn in a box around the polygon
            #  - must be offset to map back to the original location 
            r_offset, c_offset, mask = self.get_pixels_mask()
            # Map back to original location
            full_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
            full_mask[r_offset:r_offset+mask.shape[0], c_offset:c_offset+mask.shape[1]] = mask
            # Apply the mask
            img[full_mask] = fill

        # Draw the outline
        if outline is not None:
            for i in range(-1,self.shape[0]-1):
                rr, cc, val = line_aa(
                    self.v[i,0], self.v[i,1],
                    self.v[i+1,0], self.v[i+1,1]
                )
                img[rr,cc] = outline



    """
    Method to get the best colour to represent the corresponding section of an image
    """
    def get_overlapping_colour(self, img):
        # Get the mask indicating which pixels are part of the polygon
        #  - the mask is just drawn in a box around the polygon
        #  - must be offset to map back to the original location 
        r_offset, c_offset, mask = self.get_pixels_mask()
        # Map back to original location
        full_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        full_mask[r_offset:r_offset+mask.shape[0], c_offset:c_offset+mask.shape[1]] = mask

        # Get the 'average' colour of the overlapping pixels
        colour = np.mean(img[full_mask], axis=0).astype(int)
        # Get the residuals after averaging
        mse = np.square(img[full_mask] - colour).mean()
        return colour, mse


    """
    Method to draw a series of ploygon-split choices on an image
    """
    @staticmethod
    def draw_polygon_split_choices(choices, img):
        white = np.array([255,255,255])
        for split_polys in choices:
            for poly in split_polys:
                poly.draw(img,outline=white)

    """
    Static method to create a rectanngular Polygon outlining an image
    """
    @staticmethod
    def image_outline(img):
        height = img.shape[0]
        width = img.shape[1]

        outline = [
            [0,0],
            [0,width-1],
            [height-1,width-1],
            [height-1,0]
        ]
        return Polygon(np.asarray(outline))

    """
    Method to pick the best replacement polygons
    """
    def split_polygon(self, img, methods=['rectangle'], random=[False], min_area=None, min_ratio=None, rec_n=4, tri_n=3):
        assert len(methods) == len(random)
        # Get the choices for splitting this polygon 
        choices = []

        # Execute all possible / specified split functions 
        for i in range(len(methods)):
            if methods[i] == 'rectangle':
                # Only valid if the polygon is still a rectangle
                if self.v.shape[0] != 4:
                    continue
                if random[i]:
                    choices += self.random_rectangle_split(N=rec_n)
                else:
                    choices += self.grid_rectangle_split(N=rec_n)
            elif methods[i] == 'triangle':
                if random[i]:
                    choices += self.random_triangle_split(N=tri_n)
                else:
                    choices += self.grid_triangle_split(N=tri_n)
            else:
                raise ValueError("Invalid split method: %s" % methods[i])

        if not choices:
            return [], []

        # Exclude choices that include polygons that are lower than the minimum area threshold
        if min_area is not None:
            new_choices = []
            for choice in choices:
                choice_min_area = np.min([p.area() for p in choice])
                if choice_min_area > min_area:
                    new_choices.append(choice)
            choices = new_choices
        # Exclude choices that include polygons that have a width/height ratio that is lower than the threshold
        if min_ratio is not None:
            new_choices = []
            for choice in choices:
                choice_stretch_ratio = np.min([p.stretch_ratio() for p in choice])
                if choice_stretch_ratio > min_ratio:
                    new_choices.append(choice)
            choices = new_choices


        # Short-ciruit if there are no valid choices
        n_choices = len(choices)
        if n_choices == 0:
            return [], []

        # Evaluate each choice
        choice_errors = np.zeros(n_choices)
        colour_cache = []
        args = []
        for i in range(n_choices):
            args.append([choices[i], img])
        with multiprocessing.Pool(processes=4) as pool:
            results = pool.map(Polygon.evaluate_choice, args)
        for i in range(len(results)):
            choice_errors[i] = results[i][0]
            colour_cache.append(results[i][1])

        # Get the best choice based on the errors 
        best_choice = np.nanargmin(choice_errors)
        # Get corresponding polygons and colours
        return_polys = choices[best_choice]
        return_colours = colour_cache[best_choice]
        # print("Selected choice %s: %s, %s" % (best_choice, return_polys, return_colours))

        return return_polys, return_colours

    @staticmethod
    def evaluate_choice(args):
        split_polys = args[0]
        img = args[1]
        colours = []
        # Get the best colour and resulting error for each polygon
        error = 0
        for poly in split_polys:
            c, e = poly.get_overlapping_colour(img)
            error += e
            colours.append(c)

        # Take mean error
        error = error / len(split_polys)
        #print("Choice %s has average error: %s" % (i, choice_errors[i]))
        return error, colours

    def recursive_split_polygon(self, img, config, iteration=0, status={}):
        if not (config['rectangle_depth'] == 0 or config['triangle_depth'] == 0):
            raise RuntimeError("Either rectangle or triangle splits must be enabled for iteration 0")

        poly_str = str(self.v).replace('\n', ',')
        #print("*** Iteration %s on polygon %s ***" % (iteration, poly_str))
        # Stop conditions
        if iteration >= config['max_iterations']:
            return [], []

        # Get the split methods
        methods = []
        random = []
        if iteration >= config['rectangle_depth']:
            methods.append('rectangle')
            random.append(config['rectangle_method'] == 'random')
        if iteration >= config['triangle_depth']:
            methods.append('triangle')
            random.append(config['triangle_method'] == 'random')

        # Do the split 
        total_area = img.shape[0] * img.shape[1]
        threshold = int(total_area * config['area_threshold'])
        split_polys, split_colours = self.split_polygon(
            img,
            methods=methods,
            random=random,
            min_area=threshold,
            min_ratio=config['ratio_threshold'],
            rec_n=config['rec_n'],
            tri_n=config['tri_n'],
        )

        # If the split did not find any valid sub-ploygons, return now
        #  - can return a 'split' of length=1, this is not a 'valid' split
        if len(split_polys) < 2:
            return [], []

        # Call recursively on the new polygons that exceed the area threshold 
        new_polys = []
        new_colours = []
        n_successful_splits = 0
        for i in range(len(split_polys)):
            # Further split this polygon and add to list if any results are returned
            p, c = split_polys[i].recursive_split_polygon(
                img,
                config,
                iteration=iteration+1,
                status=status,
            )
            if len(p) > 1:
                new_polys += p
                new_colours += c
                n_successful_splits += len(p)
            else:
                # Failed to get a result for this sub-section, return the original
                new_polys.append(split_polys[i])
                new_colours.append(split_colours[i])

        print("Iteration %s resulted in %s splits on polygon %s " % (iteration, n_successful_splits, poly_str))
        return new_polys, new_colours


    """
    Method to get the pixels inside the Polygon
    """
    def get_pixels_mask(self):
        # Get the bounds of the polygon
        vmax = self.v.max(axis=0)
        vmin = self.v.min(axis=0)

        # Get all pixels in the bounding box (self.v is in [row,column])
        r_size = vmax[0] - vmin[0]
        c_size = vmax[1] - vmin[1]
        box_r, box_c = np.meshgrid(
            np.arange(r_size) + vmin[0],
            np.arange(c_size) + vmin[1]
        )
        box_r, box_c = box_r.flatten(), box_c.flatten()
        pixels = np.vstack((box_r,box_c)).T
        # Select only the points that are contained by the Polygon
        p = Path(self.v)
        grid = p.contains_points(pixels)
        mask = grid.reshape(c_size, r_size).T

        return vmin[0], vmin[1], mask.astype(int)


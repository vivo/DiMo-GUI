from PIL import Image, ImageDraw

class RegionTraverser:
    def __init__(self, image):
        """
        Initialize the RegionTraverser with an image.

        Parameters:
            image (PIL.Image): The input image to traverse.
        """
        self.original_image = image
        self.current_image = image.copy()
        self.image_width, self.image_height = image.size
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = self.image_width, self.image_height
        self.highlighted_image = None
        self.cropped_image = None

        self.is_mobile = self.image_width < self.image_height

    def _convert_coordinate_relative_to_region(self, coord):
        """
        Convert a coordinate from a 0-999 scale to the actual current region size scale.

        Parameters:
            coord (tuple): A tuple (x, y) representing the coordinate in the 0-999 scale.

        Returns:
            tuple: A tuple (scaled_x, scaled_y) with the coordinate scaled to the current region size.
        """
        x, y = coord
        region_width = self.x2 - self.x1
        region_height = self.y2 - self.y1

        scaled_x = self.x1 + (x / 999) * region_width
        scaled_y = self.y1 + (y / 999) * region_height

        return int(scaled_x), int(scaled_y)

    def consume_coordinate(self, x, y, new_width=None, new_height=None):
        """
        Perform a traversal on the current region using the given coordinate, relative to the current region.

        Parameters:
            x (int): The x-coordinate for centering the new region (0-999).
            y (int): The y-coordinate for centering the new region (0-999).
        """
        # Convert coordinates to the current region's scale
        x, y = self._convert_coordinate_relative_to_region((x, y))

        # Current region's width and height
        width = self.x2 - self.x1
        height = self.y2 - self.y1

        # for sspro
        if self.is_mobile:
            if new_width is None: new_width = width / 1.2
            if new_height is None: new_height = height / 2
        else:
            if new_width is None: new_width = width / 2
            if new_height is None: new_height = height / 2

        # for ss
        # if self.is_mobile:
        #     if new_width is None: new_width = width 
        #     if new_height is None: new_height = height
        # else:
        #     if new_width is None: new_width = width
        #     if new_height is None: new_height = height

        # Desired new region centered at (x, y)
        new_x1 = x - new_width / 2
        new_y1 = y - new_height / 2
        new_x2 = x + new_width / 2
        new_y2 = y + new_height / 2

        # Adjust if the new region goes out of bounds
        if new_x1 < self.x1:
            new_x1 = self.x1
            new_x2 = new_x1 + new_width
        if new_x2 > self.x2:
            new_x2 = self.x2
            new_x1 = new_x2 - new_width

        if new_y1 < self.y1:
            new_y1 = self.y1
            new_y2 = new_y1 + new_height
        if new_y2 > self.y2:
            new_y2 = self.y2
            new_y1 = new_y2 - new_height

        # Ensure coordinates are within bounds
        new_x1 = max(self.x1, new_x1)
        new_x2 = min(self.x2, new_x2)
        new_y1 = max(self.y1, new_y1)
        new_y2 = min(self.y2, new_y2)

        # Update the current region
        self.x1, self.y1, self.x2, self.y2 = map(int, (new_x1, new_y1, new_x2, new_y2))

        # Create a copy of the image and draw the rectangle and intersecting lines
        self.highlighted_image = self.original_image.copy()
        draw = ImageDraw.Draw(self.highlighted_image)

        # Draw the rectangle to highlight the region
        draw.rectangle([self.x1, self.y1, self.x2, self.y2], outline="red", width=10)

        # Calculate the midpoint of the final region
        mid_x = (self.x1 + self.x2) // 2
        mid_y = (self.y1 + self.y2) // 2

        # Draw vertical and horizontal intersecting lines
        draw.line([(mid_x, self.y1), (mid_x, self.y2)], fill="blue", width=4)  # Vertical line
        draw.line([(self.x1, mid_y), (self.x2, mid_y)], fill="blue", width=4)  # Horizontal line

        # Crop the image to the final region
        self.cropped_image = self.original_image.crop((self.x1, self.y1, self.x2, self.y2))

    def get_bounding_box(self):
        """
        Get the current bounding box coordinates.

        Returns:
            tuple: A tuple of (x1, y1, x2, y2).
        """
        return self.x1, self.y1, self.x2, self.y2

    def get_highlighted_image(self):
        """
        Get the highlighted image with the current region marked.

        Returns:
            PIL.Image: The highlighted image.
        """
        return self.highlighted_image

    def get_cropped_image(self):
        """
        Get the cropped image of the current region.

        Returns:
            PIL.Image: The cropped image.
        """
        return self.cropped_image

    def reset_region(self):
        """
        Reset the current region to the full image dimensions.
        """
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = self.image_width, self.image_height
        self.highlighted_image = None
        self.cropped_image = None
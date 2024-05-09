import math

def hexagonal_packing(W, H, D):
    # Step 1: Calculate the vertical spacing between rows
    vertical_spacing = (math.sqrt(3) / 2) * D
    
    # Step 2: Calculate the number of rows
    num_rows = math.floor(H / vertical_spacing)
    
    # Step 3: Initialize the total number of circles
    total_circles = 0
    
    # Step 4: Loop through each row to calculate the number of circles
    for row in range(1, num_rows + 1):
        if row % 2 == 1:  # Odd row
            num_circles_in_row = math.floor(W / D)
        else:  # Even row
            num_circles_in_row = math.floor((W - D / 2) / D) + 1
        
        # Add the number of circles in this row to the total
        total_circles += num_circles_in_row
    
    return total_circles

# Test the function
W = 6028  # Width of the rectangle
H = 4012   # Height of the rectangle
D = 142   # Diameter of each circle

result = hexagonal_packing(W, H, D)
print("Maximum number of circles with hexagonal packing:", result)

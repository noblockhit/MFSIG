__global__ void createLines(int *data_x, int *data_y, int *data_z,
                            int *length_arr, int* maximum_distance_arr, int*image_distance_to_pixel_factor_arr,
                            int *lines_x, int *lines_y, int *lines_z) {
    const int length = length_arr[0];
    const int maximum_distance = maximum_distance_arr[0];
    const int image_distance_to_pixel_factor = image_distance_to_pixel_factor_arr[0];

    const int thrd_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (thrd_i > length-1) {
        return;
    }
    const int x = data_x[thrd_i];
    const int y = data_y[thrd_i];
    const int z = data_z[thrd_i];
    
    int distance = 2147483647;
                    

    for (int i = 0; i < length; i++) {
        int other_x = data_x[i];
        int other_y = data_y[i];
        int other_z = data_z[i];

        int height_distance = z - other_z;
        
        if (0 < height_distance && height_distance < 3) {
            int new_distance_2d = (x-other_x)*(x-other_x) + (y-other_y)*(y-other_y);
            int new_distance_3d = new_distance_2d + (height_distance*image_distance_to_pixel_factor)*(height_distance*image_distance_to_pixel_factor);
            if (new_distance_3d < maximum_distance && new_distance_3d < distance) {
                distance = new_distance_3d;
                lines_x[thrd_i*3+1] = other_x;
                lines_y[thrd_i*3+1] = other_y;
                lines_z[thrd_i*3+1] = other_z;
            }
        }
    }
    if (lines_x[thrd_i*3+1] > -1) {
        lines_x[thrd_i*3] = x;
        lines_y[thrd_i*3] = y;
        lines_z[thrd_i*3] = z;
    }             
}
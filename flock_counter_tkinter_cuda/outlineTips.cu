__device__ int get_pos(int x, int y, int width, int color) {
    return (y * width + x) * 3 + color;
}

__device__ int get_pos_bw(int x, int y, int width) {
    return (y * width + x);
}

__device__ uint8_t get_brightness(uint8_t*img, int x, int y, int width) {
    return (img[get_pos(x, y, width, 0)] + img[get_pos(x, y, width, 1)] + img[get_pos(x, y, width, 2)]) / 3;
}

__global__ void outline_tips_method_1(uint8_t*input_image, uint8_t*output_image, int*dims, uint8_t own_thresh, uint8_t ngb_thresh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = dims[1];
    int height = dims[0];
    uint8_t outval = 255;
    if (x < width && y < height) {
        if (get_brightness(input_image, x, y, width) > own_thresh) {
            
            if (get_brightness(input_image, x-1, y, width) < ngb_thresh) {
                output_image[get_pos_bw(x, y, width)] = outval;
            } else
            if (get_brightness(input_image, x+1, y, width) < ngb_thresh) {
                output_image[get_pos_bw(x, y, width)] = outval;
            } else
            if (get_brightness(input_image, x, y-1, width) < ngb_thresh) {
                output_image[get_pos_bw(x, y, width)] = outval;
            } else
            if (get_brightness(input_image, x, y+1, width) < ngb_thresh) {
                output_image[get_pos_bw(x, y, width)] = outval;
            }
        }
    }
}

__global__ void outline_tips_method_2(uint8_t*input_image, uint8_t*output_image, int*dims) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = dims[1];
    int height = dims[0];
    int thresh = 5;
    uint8_t own_thresh = 60;
    uint8_t ngb_thresh = 70;
    uint8_t outval = 255;
    if (x < width && y < height) {
        if ((int)get_brightness(input_image, x-1, y, width)+thresh < (int)get_brightness(input_image, x, y, width)) {
            output_image[get_pos_bw(x, y, width)] = outval;
        } else
        if ((int)get_brightness(input_image, x+1, y, width)+thresh < (int)get_brightness(input_image, x, y, width)) {
            output_image[get_pos_bw(x, y, width)] = outval;
        } else
        if ((int)get_brightness(input_image, x, y-1, width)+thresh < (int)get_brightness(input_image, x, y, width)) {
            output_image[get_pos_bw(x, y, width)] = outval;
        } else
        if ((int)get_brightness(input_image, x, y+1, width)+thresh < (int)get_brightness(input_image, x, y, width)) {
            output_image[get_pos_bw(x, y, width)] = outval;
        } else if (get_brightness(input_image, x, y, width) > own_thresh) {
            
            if (get_brightness(input_image, x-1, y, width) < ngb_thresh) {
                output_image[get_pos_bw(x, y, width)] = outval;
            } else
            if (get_brightness(input_image, x+1, y, width) < ngb_thresh) {
                output_image[get_pos_bw(x, y, width)] = outval;
            } else
            if (get_brightness(input_image, x, y-1, width) < ngb_thresh) {
                output_image[get_pos_bw(x, y, width)] = outval;
            } else
            if (get_brightness(input_image, x, y+1, width) < ngb_thresh) {
                output_image[get_pos_bw(x, y, width)] = outval;
            }
        }
    }
}

__global__ void outline_tips_method_3(uint8_t*input_image, uint8_t*output_image, int*dims, int radius, double thresh) {
    int center_x = blockIdx.x * blockDim.x + threadIdx.x;
    int center_y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = dims[1];
    int height = dims[0];
    uint8_t outval = 255;

    if (center_x > width || center_y > height || center_x < 0 || center_y < 0) {
        return;
    }

    char center_b = input_image[get_pos(center_x, center_y, width, 0)];
    char center_g = input_image[get_pos(center_x, center_y, width, 1)];
    char center_r = input_image[get_pos(center_x, center_y, width, 2)];

    long delta = 0;

    int calculated_pixels = 0;

    for (int x = center_x-radius; x < center_x+radius+1; x++) {
        for (int y = center_y-radius; y < center_y+radius+1; y++) {
            if (x < 0 || y < 0 || x > width || y > height) {
                continue;
            }
            
            if (x == center_x && y == center_y) {
                continue;
            }
            
            int d = abs(center_b - input_image[get_pos(x, y, width, 0)]) + abs(center_g - input_image[get_pos(x, y, width, 1)]) + abs(center_r - input_image[get_pos(x, y, width, 2)]);
            
            delta += d;
            calculated_pixels++;
        }
    }
    double sharpness = (double)(delta) / (double)(calculated_pixels * 3 * 255);

    if (sharpness > thresh) {
        output_image[get_pos_bw(center_x, center_y, width)] = outval;
    }
}

__global__ void outline_tips_method_4(uint8_t*input_image, uint8_t*output_image, int*dims, int radius, double thresh) {
    int center_x = blockIdx.x * blockDim.x + threadIdx.x;
    int center_y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = dims[1];
    int height = dims[0];
    uint8_t outval = 255;

    if (center_x > width || center_y > height || center_x < 0 || center_y < 0) {
        return;
    }

    char center_b = input_image[get_pos(center_x, center_y, width, 0)];
    char center_g = input_image[get_pos(center_x, center_y, width, 1)];
    char center_r = input_image[get_pos(center_x, center_y, width, 2)];

    long delta = 0;

    int calculated_pixels = 0;

    for (int x = center_x-radius; x < center_x+radius+1; x++) {
        for (int y = center_y-radius; y < center_y+radius+1; y++) {
            if (x < 0 || y < 0 || x > width || y > height) {
                continue;
            }
            
            if (x == center_x && y == center_y) {
                continue;
            }
            
            int d = abs(center_b - input_image[get_pos(x, y, width, 0)]) + abs(center_g - input_image[get_pos(x, y, width, 1)]) + abs(center_r - input_image[get_pos(x, y, width, 2)]);
            
            delta += d;
            calculated_pixels++;
        }
    }
    double sharpness = (double)(delta) / (double)(calculated_pixels * 3 * 255);
    if (sharpness > thresh) {
        output_image[get_pos_bw(center_x, center_y, width)] = outval;
    }
    __syncthreads();
    int active_neighbors = 0;
    for (int i = 0; i < 4; i++) {
        for (int x = center_x-1; x < center_x+2; x++) {
            for (int y = center_y-1; y < center_y+2; y++) {
                if (x < 0 || y < 0 || x > width || y > height) {
                    continue;
                }
                
                if (x == center_x && y == center_y) {
                    continue;
                }
                
                if (output_image[get_pos_bw(x, y, width)] == outval) {
                    active_neighbors++;
                }
            }
        }
        if (active_neighbors > 3) {
            output_image[get_pos_bw(center_x, center_y, width)] = outval;
        } else {
            output_image[get_pos_bw(center_x, center_y, width)] = 0;
        }
    }    
}
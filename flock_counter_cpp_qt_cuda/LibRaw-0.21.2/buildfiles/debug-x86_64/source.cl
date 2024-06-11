typedef uchar uint8_t;

inline int get_pos(int x, int y, int width, int color) {
	return (y * width + x) * 3 + color;
}

inline uint8_t get_brightness(__global uint8_t* img, int x, int y, int width) {
	return (img[get_pos(x, y, width, 0)] + img[get_pos(x, y, width, 1)] + img[get_pos(x, y, width, 2)]) / 3;
}

__kernel void threshold_approach(__global uint8_t* input_image, __global uint8_t* output_image, int width, int height, uint8_t own_thresh, uint8_t ngb_thresh) {
	int x = get_global_id(0);
	int y = get_global_id(1);

	// Check if the (x, y) is within the image
	if (x >= width || y >= height) {
		return;
	}

	uint8_t outval = 255;

	if (get_brightness(input_image, x, y, width) > own_thresh) {
		if (get_brightness(input_image, x - 1, y, width) < ngb_thresh) {
			output_image[get_pos(x, y, width, 0)] = outval;
			output_image[get_pos(x, y, width, 1)] = outval;
			output_image[get_pos(x, y, width, 2)] = outval;
		} else
		if (get_brightness(input_image, x + 1, y, width) < ngb_thresh) {
			output_image[get_pos(x, y, width, 0)] = outval;
			output_image[get_pos(x, y, width, 1)] = outval;
			output_image[get_pos(x, y, width, 2)] = outval;
		} else
		if (get_brightness(input_image, x, y - 1, width) < ngb_thresh) {
			output_image[get_pos(x, y, width, 0)] = outval;
			output_image[get_pos(x, y, width, 1)] = outval;
			output_image[get_pos(x, y, width, 2)] = outval;
		} else
		if (get_brightness(input_image, x, y + 1, width) < ngb_thresh) {
			output_image[get_pos(x, y, width, 0)] = outval;
			output_image[get_pos(x, y, width, 1)] = outval;
			output_image[get_pos(x, y, width, 2)] = outval;
		}
		else {
			output_image[get_pos(x, y, width, 0)] = 0;
			output_image[get_pos(x, y, width, 1)] = 0;
			output_image[get_pos(x, y, width, 2)] = 0;
		}
	}
}

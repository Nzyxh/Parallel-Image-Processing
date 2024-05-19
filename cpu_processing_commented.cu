#include "header.h"

#define TILE_SIZE 16
#define RADIUS 2
#define TILE_WIDTH (TILE_SIZE + 2 * RADIUS)

// to add padding to the image, we need to add 2 pixels on each side


// filters are stored in constant memory for fast access on the device
__constant__ float d_filters[3][25]; 

// filters are initialized here only once on the host and copied to the device
//in constant memory

void initializeFilters() {
    float h_filters[3][25] = {
        {
            -1 / 8.0f, -1 / 8.0f, -1 / 8.0f, -1 / 8.0f, -1 / 8.0f,
            -1 / 8.0f, 2 / 8.0f, 2 / 8.0f, 2 / 8.0f, -1 / 8.0f,
            -1 / 8.0f, 2 / 8.0f, 8 / 8.0f, 2 / 8.0f, -1 / 8.0f,
            -1 / 8.0f, 2 / 8.0f, 2 / 8.0f, 2 / 8.0f, -1 / 8.0f,
            -1 / 8.0f, -1 / 8.0f, -1 / 8.0f, -1 / 8.0f, -1 / 8.0f
        },
        {
            1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f,
            4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
            7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f,
            4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
            1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f
        },
        {
            -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f,
            -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f,
            -1 / 24.0f, -1 / 24.0f, 24 / 24.0f, -1 / 24.0f, -1 / 24.0f,
            -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f,
            -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f
        }
    };

    cudaMemcpyToSymbol(d_filters, h_filters, sizeof(h_filters));
}



void PictureHost_FILTER(png_byte *h_In, png_byte *h_Out, int h, int w, float *h_filt)
{
  float out;
  png_byte b;

  for (int Row = 2; Row < h - 2; Row++)
    for (int Col = 2; Col < w - 2; Col++)
    {
      for (int color = 0; color < 3; color++)
      {
        out = 0;
        for (int i = -2; i <= 2; i++)
          for (int j = -2; j <= 2; j++)
            out += h_filt[(i + 2) * 5 + j + 2] * h_In[((Row + i) * w + (Col + j)) * 3 + color];
        b = (png_byte)fminf(fmaxf(out, 0.0f), 255.0f);
        h_Out[(Row * w + Col) * 3 + color] = b;
      }
    }
}


void execute_jobs_cpu(PROCESSING_JOB **jobs)
{
  int count = 0;
  float *h_filter;
  while (jobs[count] != NULL)
  {
    printf("Processing job: %s -> %s -> %s\n", jobs[count]->source_name, getStrAlgoFilterByType(jobs[count]->processing_algo), jobs[count]->dest_name);

    h_filter = getAlgoFilterByType(jobs[count]->processing_algo);
    PictureHost_FILTER(jobs[count]->source_raw, jobs[count]->dest_raw,
                       jobs[count]->height, jobs[count]->width, h_filter);
    count++;
  }
}

// this method applies the filter to the image on the device

__global__ void PictureDevice_FILTER(png_byte *d_In, png_byte *d_Out, int height, int width, int filterIndex) {
    // check part 2 for the explanation of the PictureDevice_FILTER function
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH][3]; 
    

// check part 3 for the explanation of the PictureDevice_FILTER function
    int Col = blockIdx.x * TILE_SIZE + threadIdx.x - RADIUS;
    int Row = blockIdx.y * TILE_SIZE + threadIdx.y - RADIUS;

// check part 4 for the explanation of the PictureDevice_FILTER function
    if (Row >= 0 && Row < height && Col >= 0 && Col < width) {
        for (int color = 0; color < 3; color++) {
            tile[threadIdx.y][threadIdx.x][color] = d_In[(Row * width + Col) * 3 + color];
        }
    } else {
        for (int color = 0; color < 3; color++) {
            tile[threadIdx.y][threadIdx.x][color] = 0.0f; // Handle border cases by setting them to 0
        }
    }
// check part 5 for the explanation of the PictureDevice_FILTER function
    __syncthreads();
// check part 6 for the explanation of the PictureDevice_FILTER function
    if (threadIdx.y >= RADIUS && threadIdx.y < TILE_SIZE + RADIUS &&
        threadIdx.x >= RADIUS && threadIdx.x < TILE_SIZE + RADIUS &&
        Row >= 2 && Row < height - 2 && Col >= 2 && Col < width - 2) {

// check part 7 for the explanation of the PictureDevice_FILTER function
        
        for (int color = 0; color < 3; color++) {
            float out = 0.0f;
            for (int i = -RADIUS; i <= RADIUS; i++) {
                for (int j = -RADIUS; j <= RADIUS; j++) {
                    out += d_filters[filterIndex][(i + RADIUS) * 5 + (j + RADIUS)] * tile[threadIdx.y + i][threadIdx.x + j][color];
                }
            }
            png_byte b = (png_byte)fminf(fmaxf(out, 0.0f), 255.0f);
            d_Out[(Row * width + Col) * 3 + color] = b;
        }
    }
}

PictureHost_FILTER FUNCTION EXPLAINED 

1) __global__ void PictureDevice_FILTER(png_byte *d_In, png_byte *d_Out, int height, int width, int filterIndex)

This is the CUDA kernel function that runs on the GPU. 
It applies a convolution filter to an input image.
d_In is a pointer to the input image data in device memory.
d_Out is a pointer to the output image data in device memory.
height and width specify the dimensions of the image.
filterIndex specifies which filter to use 
from a set of pre-defined filters stored in constant memory.

2) Shared Memory Declaration:

__shared__ float tile[TILE_WIDTH][TILE_WIDTH][3];

Allocates shared memory for the tile, including the halo region.
TILE_WIDTH is the size of the tile including the border (halo). 
The third dimension [3] is for the RGB color channels.

3) Calculate Global Indices:

int Col = blockIdx.x * TILE_SIZE + threadIdx.x - RADIUS;
int Row = blockIdx.y * TILE_SIZE + threadIdx.y - RADIUS;

Calculates the global column (Col) and row (Row) indices for the current thread.
blockIdx.x and blockIdx.y specify the block's position within the grid.
threadIdx.x and threadIdx.y specify the thread's position within the block.
RADIUS accounts for the padding required by the convolution filter.

4) Load Data into Shared Memory:

if (Row >= 0 && Row < height && Col >= 0 && Col < width) {
    for (int color = 0; color < 3; color++) {
        tile[threadIdx.y][threadIdx.x][color] = d_In[(Row * width + Col) * 3 + color];
    }
} else {
    for (int color = 0; color < 3; color++) {
        tile[threadIdx.y][threadIdx.x][color] = 0.0f;
    }
}
Each thread loads a pixel from the input image into shared memory.
If the pixel is within the image boundaries (Row and Col are valid), 
the pixel data is loaded into the shared memory tile.
If the pixel is outside the image boundaries, it sets the value to 0.0f to handle border cases.


5) Synchronize Threads:

__syncthreads();

Ensures all threads in the block have finished loading their pixels 
into shared memory before any thread proceeds to the convolution computation.

6) Check Valid Processing Region:

if (threadIdx.y >= RADIUS && threadIdx.y < TILE_SIZE + RADIUS &&
    threadIdx.x >= RADIUS && threadIdx.x < TILE_SIZE + RADIUS &&
    Row >= 2 && Row < height - 2 && Col >= 2 && Col < width - 2)

Ensures the thread is within 
the central region of the tile (excluding the halo) and within valid image boundaries.
This avoids processing pixels 
at the very edges of the image where the convolution filter cannot be fully applied.

7) Apply Convolution Filter:

for (int color = 0; color < 3; color++) {
    float out = 0.0f;
    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            out += d_filters[filterIndex][(i + RADIUS) * 5 + (j + RADIUS)] * tile[threadIdx.y + i][threadIdx.x + j][color];
        }
    }
    png_byte b = (png_byte)fminf(fmaxf(out, 0.0f), 255.0f);
    d_Out[(Row * width + Col) * 3 + color] = b;
}

For each color channel (RGB), the thread applies the convolution 
filter to the current pixel using the values stored in shared memory.
The convolution filter is applied by iterating over the filter window 
centered on the current pixel.
The output value is clamped to the range [0, 255] 
to ensure valid pixel values and cast to png_byte.
The result is written to the corresponding position in the output image.




void execute_jobs_gpu(PROCESSING_JOB **jobs) {
    int count = 0;  // Initialize job counter
    png_byte *d_In = nullptr, *d_Out = nullptr;  // Pointers for input and output image data on the device
    size_t maxNumPixels = 0;  // Variable to store the maximum number of pixels

    // Determine the maximum number of pixels to allocate memory once
    while (jobs[count] != NULL) {
        size_t numPixels = jobs[count]->height * jobs[count]->width * 3; // 3 for RGB channels
        if (numPixels > maxNumPixels) {
            maxNumPixels = numPixels;  // Update maxNumPixels if the current job has more pixels
        }
        count++;
    }

    // Allocate memory on the device
    // Memory is allocated once here based on the largest job and reused for all jobs
    cudaMalloc((void **)&d_In, maxNumPixels * sizeof(png_byte));  // Allocate memory for the input image
    cudaMalloc((void **)&d_Out, maxNumPixels * sizeof(png_byte));  // Allocate memory for the output image

    // Initialize filters once
    initializeFilters();  // Initialize the convolution filters

    count = 0;  // Reset the job counter
    char* previous_source_name = nullptr;  // Initialize the previous source name

    do {
        // Print information about the current job
        printf("Processing job: %s -> %s -> %s\n", jobs[count]->source_name, getStrAlgoFilterByType(jobs[count]->processing_algo), jobs[count]->dest_name);

        size_t numPixels = jobs[count]->height * jobs[count]->width * 3; // 3 for RGB channels

        // Check if the previous image is the same as the current image
        // this is done to avoid unnecessary data transfers between host and device
        if (previous_source_name == nullptr || strcmp(previous_source_name, jobs[count]->source_name) != 0) {
            // Copy the new image data from host to device
            cudaMemcpy(d_In, jobs[count]->source_raw, numPixels * sizeof(png_byte), cudaMemcpyHostToDevice);
            previous_source_name = jobs[count]->source_name;  // Update the previous source name
        }

        // Calculate the grid and block dimensions
        dim3 blocks((jobs[count]->width + TILE_SIZE - 1) / TILE_SIZE, (jobs[count]->height + TILE_SIZE - 1) / TILE_SIZE);
        dim3 threads(TILE_WIDTH, TILE_WIDTH);

        // Launch the kernel with filter index
        int filterIndex = jobs[count]->processing_algo; // Assuming the enum values match the filter array index
        PictureDevice_FILTER<<<blocks, threads>>>(d_In, d_Out, jobs[count]->height, jobs[count]->width, filterIndex);

        // Copy the result back to the host
        cudaMemcpy(jobs[count]->dest_raw, d_Out, numPixels * sizeof(png_byte), cudaMemcpyDeviceToHost);

        count++;  // Increment the job counter
    } while (jobs[count] != NULL);  // Continue until all jobs are processed

    // Free device memory
    cudaFree(d_In);  // Free input image memory
    cudaFree(d_Out);  // Free output image memory
}


1) Function declaration:

void execute_jobs_gpu(PROCESSING_JOB **jobs)

This function executes image processing jobs on the GPU.
jobs is an array of pointers to PROCESSING_JOB structures, each representing an image processing task.

2) Variable Initialization:

int count = 0;
png_byte *d_In = nullptr, *d_Out = nullptr;
size_t maxNumPixels = 0;

Initializes variables:
count is used to iterate through the jobs.
d_In and d_Out are pointers to the input and output image data in device memory.
maxNumPixels stores the maximum number of pixels in any job's image.

3) Determine Maximum Number of Pixels:

while (jobs[count] != NULL) {
    size_t numPixels = jobs[count]->height * jobs[count]->width * 3; // 3 for RGB channels
    if (numPixels > maxNumPixels) {
        maxNumPixels = numPixels;
    }
    count++;
}

Iterates through the jobs to find the maximum number of pixels in any job's image.
This is used to allocate sufficient memory on the GPU.


4) Allocate Device Memory:

cudaMalloc((void **)&d_In, maxNumPixels * sizeof(png_byte));
cudaMalloc((void **)&d_Out, maxNumPixels * sizeof(png_byte));

Allocates memory on the GPU for the input and output image data. 
The amount of memory allocated is based on the maximum number of pixels found in the previous step.

5) Initialize Filters:

initializeFilters();
Initializes the convolution filters.
This function is assumed to copy filter coefficients to the GPUs constant memory.

6) Reset Counter and Initialize Previous Source Name:

count = 0;
char* previous_source_name = nullptr;

Resets the count variable to start processing jobs from the beginning.
Initializes previous_source_name to keep track of the last processed image.

7) Process Each Job:

do {
    printf("Processing job: %s -> %s -> %s\n", jobs[count]->source_name, getStrAlgoFilterByType(jobs[count]->processing_algo), jobs[count]->dest_name);

    size_t numPixels = jobs[count]->height * jobs[count]->width * 3; // 3 for RGB channels

    // Check if the previous image is the same as the current image
    if (previous_source_name == nullptr || strcmp(previous_source_name, jobs[count]->source_name) != 0) {
        cudaMemcpy(d_In, jobs[count]->source_raw, numPixels * sizeof(png_byte), cudaMemcpyHostToDevice);
        previous_source_name = jobs[count]->source_name;
    }

    dim3 blocks((jobs[count]->width + TILE_SIZE - 1) / TILE_SIZE, (jobs[count]->height + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threads(TILE_WIDTH, TILE_WIDTH);

    // Launch the kernel with filter index
    int filterIndex = jobs[count]->processing_algo; // Assuming the enum values match the filter array index
    PictureDevice_FILTER<<<blocks, threads>>>(d_In, d_Out, jobs[count]->height, jobs[count]->width, filterIndex);

    // Copy result back to host
    cudaMemcpy(jobs[count]->dest_raw, d_Out, numPixels * sizeof(png_byte), cudaMemcpyDeviceToHost);

    count++;
} while (jobs[count] != NULL);


Iterates through each job:
Prints information about the current job being processed.
Calculates the number of pixels in the current job's image.
If the current image is different from the previous one, it copies the new image data from host to device.
Sets up the grid and block dimensions for the kernel launch.
Launches the PictureDevice_FILTER kernel with the appropriate filter index.
Copies the processed image data from device to host.
Increments the job counter.

8) Free Device Memory:

cudaFree(d_In);
cudaFree(d_Out);
Frees the allocated memory on the GPU for the input and output images.

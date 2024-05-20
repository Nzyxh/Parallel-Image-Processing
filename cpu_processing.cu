#include "header.h"

#define TILE_SIZE 16
#define RADIUS 2
#define TILE_WIDTH (TILE_SIZE + 2 * RADIUS)



__constant__ float d_filters[3][25]; 



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
            -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f,
            -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f,
            -1 / 24.0f, -1 / 24.0f, 24 / 24.0f, -1 / 24.0f, -1 / 24.0f,
            -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f,
            -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f
        },
        {
             1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f,
            4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
            7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f,
            4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
            1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f
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

__global__ void PictureDevice_FILTER(png_byte *d_In, png_byte *d_Out, int height, int width, int filterIndex) {
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH][3]; 
    


    int Col = blockIdx.x * TILE_SIZE + threadIdx.x - RADIUS;
    int Row = blockIdx.y * TILE_SIZE + threadIdx.y - RADIUS;

    if (Row >= 0 && Row < height && Col >= 0 && Col < width) {
        for (int color = 0; color < 3; color++) {
            tile[threadIdx.y][threadIdx.x][color] = d_In[(Row * width + Col) * 3 + color];
        }
    } else {
        for (int color = 0; color < 3; color++) {
            tile[threadIdx.y][threadIdx.x][color] = 0.0f; 
        }
    }
    __syncthreads();

    if (threadIdx.y >= RADIUS && threadIdx.y < TILE_SIZE + RADIUS &&
        threadIdx.x >= RADIUS && threadIdx.x < TILE_SIZE + RADIUS &&
        Row >= 2 && Row < height - 2 && Col >= 2 && Col < width - 2) {
        
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



void execute_jobs_gpu(PROCESSING_JOB **jobs) {
    int count = 0;
    png_byte *d_In = nullptr, *d_Out = nullptr;
    size_t maxNumPixels = 0;

    while (jobs[count] != NULL) {
        size_t numPixels = jobs[count]->height * jobs[count]->width * 3; // 3 for RGB channels
        if (numPixels > maxNumPixels) {
            maxNumPixels = numPixels;
        }
        count++;
    }


    cudaMalloc((void **)&d_In, maxNumPixels * sizeof(png_byte));
    cudaMalloc((void **)&d_Out, maxNumPixels * sizeof(png_byte));


    initializeFilters();

    count = 0;
    char* previous_source_name = nullptr;

    do {
        printf("Processing job: %s -> %s -> %s\n", jobs[count]->source_name, getStrAlgoFilterByType(jobs[count]->processing_algo), jobs[count]->dest_name);

        size_t numPixels = jobs[count]->height * jobs[count]->width * 3; 


        if (previous_source_name == nullptr || strcmp(previous_source_name, jobs[count]->source_name) != 0) {
            cudaMemcpy(d_In, jobs[count]->source_raw, numPixels * sizeof(png_byte), cudaMemcpyHostToDevice);
            previous_source_name = jobs[count]->source_name;
        }

        dim3 blocks((jobs[count]->width + TILE_SIZE - 1) / TILE_SIZE, (jobs[count]->height + TILE_SIZE - 1) / TILE_SIZE);
        dim3 threads(TILE_WIDTH, TILE_WIDTH);


        int filterIndex = jobs[count]->processing_algo; 
        PictureDevice_FILTER<<<blocks, threads>>>(d_In, d_Out, jobs[count]->height, jobs[count]->width, filterIndex);


        cudaMemcpy(jobs[count]->dest_raw, d_Out, numPixels * sizeof(png_byte), cudaMemcpyDeviceToHost);

        count++;
    } while (jobs[count] != NULL);

    cudaFree(d_In);
    cudaFree(d_Out);
}
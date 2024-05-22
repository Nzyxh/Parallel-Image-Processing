#include "header.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void dummyKernel() {
    //Empty kernel to activate GPU
}


void activateGPU() {
    dummyKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
typedef struct {
    char source[256];
    char operation[256];
    char destination[256];
} Job;

Job* readJobs(const char* filename, int* jobCount) {
    FILE* infile = fopen(filename, "r");
    if (!infile) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    int capacity = 10;
    Job* jobs = malloc(capacity * sizeof(Job));
    if (!jobs) {
        perror("Error allocating memory");
        exit(EXIT_FAILURE);
    }

    char line[768];
    *jobCount = 0;
    while (fgets(line, sizeof(line), infile)) {
        if (*jobCount >= capacity) {
            capacity *= 2;
            jobs = realloc(jobs, capacity * sizeof(Job));
            if (!jobs) {
                perror("Error reallocating memory");
                exit(EXIT_FAILURE);
            }
        }
        sscanf(line, "%s %s %s", jobs[*jobCount].source, jobs[*jobCount].operation, jobs[*jobCount].destination);
        (*jobCount)++;
    }
    fclose(infile);

    return jobs;
}

int compareJobs(const void* a, const void* b) {
    Job* jobA = (Job*)a;
    Job* jobB = (Job*)b;
    return strcmp(jobA->source, jobB->source);
}

void sortJobs(Job* jobs, int jobCount) {
    qsort(jobs, jobCount, sizeof(Job), compareJobs);
}

void writeJobs(const char* filename, Job* jobs, int jobCount) {
    FILE* outfile = fopen(filename, "w");
    if (!outfile) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < jobCount; i++) {
        fprintf(outfile, "%s %s %s\n", jobs[i].source, jobs[i].operation, jobs[i].destination);
    }
    fclose(outfile);
}


long long timeInMilliseconds(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (((long long)tv.tv_sec) * 1000) + (tv.tv_usec / 1000);
}


int main(int argc, char **argv)
{


    activateGPU();

    long long processing_start = timeInMilliseconds();
    PROCESSING_JOB **jobs = prepare_jobs(argv[1]);
    long long processing_end = timeInMilliseconds();



    long long gpu_start = timeInMilliseconds();
    execute_jobs_gpu(jobs);
    long long gpu_end = timeInMilliseconds();

    printf("Execution Time of processing part: %lld ms\n",gpu_end-gpu_start);



    long long write_start = timeInMilliseconds();
    write_jobs_output_files(jobs);
    long long write_end = timeInMilliseconds();


}
#include <iostream>
#include <fstream>
#include <cmath>
#include <curand_kernel.h>

#define IMAGE_DIMENSION 500

void writeImgArrToFile(int *arr, std::string fileName)
{
    std::ofstream arrayFile(fileName);

    for (int i = 0; i < IMAGE_DIMENSION * IMAGE_DIMENSION; i++)
    {
        if (i % IMAGE_DIMENSION == IMAGE_DIMENSION - 1)
        {
            arrayFile << arr[i] << ';';
        }
        else
        {
            arrayFile << arr[i] << ',';
        }
    }
}

__global__ void generateImage(int *pixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < IMAGE_DIMENSION * IMAGE_DIMENSION)
    {
        pixels[i] = i % IMAGE_DIMENSION;
    }
}

int main()
{
    int *pixels;
    cudaMallocManaged(&pixels, IMAGE_DIMENSION * IMAGE_DIMENSION * sizeof(int));

    int blockSize = 256;
    int numBlocks = std::ceil(IMAGE_DIMENSION * IMAGE_DIMENSION / (float)blockSize);

    std::cout << "Starting " << numBlocks << " blocks of size " << blockSize << std::endl;
    std::cout << numBlocks * blockSize << " threads in total for ";
    std::cout << IMAGE_DIMENSION * IMAGE_DIMENSION << " pixels" << std::endl;

    generateImage<<<numBlocks, blockSize>>>(pixels);

    cudaDeviceSynchronize();

    writeImgArrToFile(pixels, "img.txt");

    cudaFree(pixels);
    return 0;
}
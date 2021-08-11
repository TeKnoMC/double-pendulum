#include <iostream>
#include <fstream>
#include <cmath>

#define IMAGE_DIMENSION 2000
#define G 9.81
#define PI 3.14159265358979323846

#define ARRAY_LENGTH 3000
#define MAX_TIME 30

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

struct Vector4
{
    double d1, d2, d3, d4;
    __device__ Vector4(double d1in, double d2in, double d3in, double d4in)
    {
        d1 = d1in;
        d2 = d2in;
        d3 = d3in;
        d4 = d4in;
    };

    __device__ Vector4()
    {
        d1 = 0;
        d2 = 0;
        d3 = 0;
        d4 = 0;
    };

    __device__ Vector4 operator+(const Vector4 v)
    {
        return Vector4(d1 + v.d1, d2 + v.d2, d3 + v.d3, d4 + v.d4);
    };

    __device__ Vector4 operator*(const double d)
    {
        return Vector4(d1 * d, d2 * d, d3 * d, d4 * d);
    };
};

struct Coordinate
{
    double x, y;
    __device__ Coordinate(double xin, double yin)
    {
        x = xin;
        y = yin;
    };
};

__device__ void linspace(double *arr, double start, double stop, int arrLength)
{
    double delta = (stop - start) / (double) arrLength;

    arr[0] = start;
    for (int i = 1; i < arrLength; i += 1)
    {
        arr[i] = arr[i-1] + delta;
    }
}

__device__ Vector4 deriv(Vector4 y0, double t, double *args)
{
    double l1 = args[0];
    double l2 = args[1];
    double m1 = args[2];
    double m2 = args[3];

    double theta1 = y0.d1;
    double theta2 = y0.d2;
    double omega1 = y0.d3;
    double omega2 = y0.d4;

    double sinVal = sin(theta1 - theta2);
    double cosVal = cos(theta1 - theta2);

    double d1 = l1 * (m1 + m2 * sinVal * sinVal);
    double d2 = l2 * (m1 + m2 * sinVal * sinVal);

    double n1 = (m2 * G * sin(theta2) * cosVal
                - m2 * sinVal * (l1 * omega1 * omega1 * cosVal + l2 * omega2 * omega2)
                - (m1 + m2) * G * std::sin(theta1));

    double n2 = ((m1 + m2) * (l1 * omega1 * omega1 * sinVal
                - G * sin(theta2) + G * sin(theta1) * cosVal)
                + m2 * l2 * omega2 * omega2 * sinVal * cosVal);

    double omega1Dot = n1 / d1;
    double omega2Dot = n2 / d2;

    return Vector4(omega1, omega2, omega1Dot, omega2Dot);
}

__device__ void rk4Solve(Vector4 *resultVector, Vector4 y0, double *args)
{
    double tArr[ARRAY_LENGTH] = {0};
    linspace(tArr, 0, MAX_TIME, ARRAY_LENGTH);

    double h = tArr[1] - tArr[0];

    Vector4 previousY = y0;

    resultVector[0] = y0;

    for (int i = 0; i < ARRAY_LENGTH; i++)
    {
        Vector4 k1 = deriv(previousY, tArr[i], args);
        Vector4 k2 = deriv(previousY + k1 * (h / 2.0), tArr[i] + h / 2.0, args);
        Vector4 k3 = deriv(previousY + k2 * (h / 2.0), tArr[i] + h / 2.0, args);
        Vector4 k4 = deriv(previousY + k3 * h, tArr[i] + h, args);

        Vector4 y = previousY + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0);
        resultVector[i] = y;
        previousY = y;
    }
}

__device__ int getChaosRating(Vector4 y0, double *args)
{
    Vector4 y[ARRAY_LENGTH] = {Vector4()};
    rk4Solve(y, y0, args);

    for (int i = 0; i < ARRAY_LENGTH; i++)
    {
        if (y[i].d1 > PI || y[i].d1 < -PI || y[i].d2 > PI || y[i].d2 < -PI)
        {
            return i;
        }
    }

    return ARRAY_LENGTH;
}

__device__ Coordinate mapIdxToRange(int i, int j, double xmin, double xmax,
                            double ymin, double ymax, int length)
{
    return Coordinate(
        ((double)i / ((double)length - 1.0)) * (xmax - xmin) + xmin,
        ((double)j / ((double)length - 1.0)) * (ymax - ymin) + ymin
    );
}

__global__ void generateImage(int *pixels)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < IMAGE_DIMENSION * IMAGE_DIMENSION;
         i += blockDim.x * gridDim.x)
    {
        int xIdx = i % IMAGE_DIMENSION;
        int yIdx = i / IMAGE_DIMENSION;

        Coordinate coord = mapIdxToRange(xIdx, yIdx, -PI, PI, -PI, PI, IMAGE_DIMENSION);
        Vector4 y0 = Vector4(coord.x, coord.y, 0, 0);

        double args[4] = {1, 1, 1, 1};  // (L1, L2, M1, M2)

        pixels[i] = getChaosRating(y0, args);
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

    std::cout << "Writing to file..." << std::endl;

    writeImgArrToFile(pixels, "img.txt");

    cudaFree(pixels);
    return 0;
}
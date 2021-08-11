#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#define IMAGE_DIMENSION 1000
#define G 9.81
#define PI 3.14159265358979323846

void write2dArrayToFile(int arr[IMAGE_DIMENSION][IMAGE_DIMENSION], std::string fileName)
{
    std::ofstream arrayFile(fileName);

    for (int i = 0; i < IMAGE_DIMENSION; i++)
    {
        for (int j = 0; j < IMAGE_DIMENSION; j++)
        {
            arrayFile << arr[i][j] << ',';
        }
        arrayFile << ';';
    }
}

struct Vector4
{
    double d1, d2, d3, d4;
    Vector4(double d1in, double d2in, double d3in, double d4in)
    {
        d1 = d1in;
        d2 = d2in;
        d3 = d3in;
        d4 = d4in;
    };

    Vector4()
    {
        d1 = 0;
        d2 = 0;
        d3 = 0;
        d4 = 0;
    };

    
    Vector4 operator+(const Vector4 v)
    {
        return Vector4(d1 + v.d1, d2 + v.d2, d3 + v.d3, d4 + v.d4);
    };

    Vector4 operator*(const double d)
    {
        return Vector4(d1 * d, d2 * d, d3 * d, d4 * d);
    };
};

struct Coordinate
{
    double x, y;
    Coordinate(double xin, double yin)
    {
        x = xin;
        y = yin;
    };
};

double *linspace(double start, double stop, int arrLength)
{
    double delta = (stop - start) / (double) arrLength;

    double *arr = new double[arrLength];
    arr[0] = start;

    for (int i = 1; i < arrLength; i += 1)
    {
        arr[i] = arr[i-1] + delta;
    }

    return arr;
}

Vector4 deriv(Vector4 y0, double t, double *args)
{
    double l1 = args[0];
    double l2 = args[1];
    double m1 = args[2];
    double m2 = args[3];

    double theta1 = y0.d1;
    double theta2 = y0.d2;
    double omega1 = y0.d3;
    double omega2 = y0.d4;

    double sin = std::sin(theta1 - theta2);
    double cos = std::cos(theta1 - theta2);

    double d1 = l1 * (m1 + m2 * sin * sin);
    double d2 = l2 * (m1 + m2 * sin * sin);

    double n1 = (m2 * G * std::sin(theta2) * cos
                - m2 * sin * (l1 * omega1 * omega1 * cos + l2 * omega2 * omega2)
                - (m1 + m2) * G * std::sin(theta1));

    double n2 = ((m1 + m2) * (l1 * omega1 * omega1 * sin
                - G * std::sin(theta2) + G * std::sin(theta1) * cos)
                + m2 * l2 * omega2 * omega2 * sin * cos);

    double omega1Dot = n1 / d1;
    double omega2Dot = n2 / d2;

    return Vector4(omega1, omega2, omega1Dot, omega2Dot);
}

Vector4 *rk4Solve(Vector4 y0, int arrLength, double tmax, double *args)
{
    double *tArr = linspace(0, tmax, arrLength);
    double h = tArr[1] - tArr[0];

    Vector4 previousY = y0;

    Vector4 *solution = new Vector4[arrLength];
    solution[0] = y0;

    for (int i = 0; i < arrLength; i++)
    {
        Vector4 k1 = deriv(previousY, tArr[i], args);
        Vector4 k2 = deriv(previousY + k1 * (h / 2.0), tArr[i] + h / 2.0, args);
        Vector4 k3 = deriv(previousY + k2 * (h / 2.0), tArr[i] + h / 2.0, args);
        Vector4 k4 = deriv(previousY + k3 * h, tArr[i] + h, args);

        Vector4 y = previousY + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0);
        solution[i] = y;
        previousY = y;
    }

    return solution;
}

int getChaosRating(Vector4 y0, double *args)
{
    int arrLength = 3000;
    Vector4 *y = rk4Solve(y0, arrLength, 30, args);

    for (int i = 0; i < arrLength; i++)
    {
        if (std::abs(y[i].d1) > PI || std::abs(y[i].d2) > PI)
        {
            return i;
        }
    }

    return arrLength;
}

Coordinate mapIdxToRange(int i, int j, double xmin, double xmax,
                            double ymin, double ymax, int length)
{
    return Coordinate(
        ((double)i / ((double)length - 1.0)) * (xmax - xmin) + xmin,
        ((double)j / ((double)length - 1.0)) * (ymax - ymin) + ymin
    );
}

void generateImage(int imgArr[IMAGE_DIMENSION][IMAGE_DIMENSION],double xmin,
                        double xmax, double ymin, double ymax, double *args)
{
    for (int i = 0; i < IMAGE_DIMENSION; i++)
    {
        for (int j = 0; j < IMAGE_DIMENSION; j++)
        {
            Coordinate coord = mapIdxToRange(j, i, xmin, xmax, ymin, ymax, IMAGE_DIMENSION);
            Vector4 y0 = Vector4(coord.x, coord.y, 0, 0);
            imgArr[i][j] = getChaosRating(y0, args);
        }

        std::cout << "Row " << i << " of " << IMAGE_DIMENSION << std::endl;
    }
}

int main()
{
    std::cout << "Beginning image generation..." << std::endl;
    
    auto imgArr = new int[IMAGE_DIMENSION][IMAGE_DIMENSION];

    //cudaMallocManaged(&imgArr, IMAGE_DIMENSION * IMAGE_DIMENSION * sizeof(int));

    double args[4] = {1.0, 1.0, 1.0, 1.0};

    generateImage(imgArr, -PI, PI, -PI, PI, args);

    write2dArrayToFile(imgArr, "1000-test.txt");

    delete[] imgArr;

    return 0;
}
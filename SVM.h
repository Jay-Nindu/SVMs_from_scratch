#pragma once
#include <functional>
#include <vector>
#include "matrix.h"

enum class Kernel : char {LINEAR, RBF};

struct SVM{
    int numFeatures;
    double slackC, bias, tolerance, threshold, gamma;
    Kernel kern;
    std::vector<matrix> samples;
    std::vector<matrix> labels; //need to rewrite this
    std::vector<double> errors;
    matrix hyperplane;
    std::vector<double> lagrangianMultipliers;


    SVM(std::vector<matrix> & x, std::vector<matrix> & y, double C, double tol, Kernel k, double sd);
    void SMO();
    std::function<double(matrix&, matrix&)> computeKernel;
    bool examineSample(int i2);
    bool takeStep (int i1, int i2);

    double predict(int i1);
    double predict(matrix i);
};
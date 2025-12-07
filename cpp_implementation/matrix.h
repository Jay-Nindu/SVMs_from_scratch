#pragma once
#include<vector>
#include <iostream>


struct matrix{
    int rows, columns;
    std::vector<std::vector<double>> values;

    matrix(int r=0, int c=0);
    matrix(int r, int c, double val);
    matrix(matrix &&other) noexcept; 

    matrix(const matrix& other);

    void clear();
    std::vector<double>& operator[] (int idx);
    const std::vector<double>& operator[](int idx) const;

    matrix transpose() const;

    matrix& operator=(matrix rhs);

    friend void swap(matrix &lhs, matrix &rhs);
    matrix& operator*=(const double scalar);
    friend matrix operator*(const matrix &lhs, const double scalar);

    matrix& operator/=(const double scalar);
    friend matrix operator/(const matrix &lhs, const double scalar);

    friend matrix operator*(const matrix &lhs, const matrix& rhs);
    matrix& operator*=(const matrix& rhs);

    matrix& operator+=(const matrix&rhs);
    friend matrix operator+(const matrix &lhs, const matrix&rhs);

    matrix& operator-=(const matrix&rhs);
    friend matrix operator-(const matrix &lhs, const matrix&rhs);

    friend bool operator<(const matrix& lhs, const matrix& rhs);
    friend matrix elementWiseDiv(const matrix& lhs, const matrix& rhs);
    friend matrix elementWiseMult(const matrix& lhs, const matrix& rhs);
    friend matrix sqrt(const matrix& lhs);

    double normalise();
};

std::ostream& operator<<(std::ostream& os, const matrix& obj);
std::istream& operator>>(std::istream& is, matrix& obj);

#include "matrix.h"
#include <random>
#include<math.h>

using namespace std;

matrix::matrix(int r, int c){
    rows = r;
    columns = c;
    values.resize(rows, vector<double>(columns));
    // note: rxc = num_inputs x numoutputs

    random_device seed; //uniform distribution rng - used to get seed for mt199937 rg
    mt19937 gen{seed()}; //uniform rng
    normal_distribution normal{0.0, sqrt(2.0/rows)}; //converts uniform rng numbers to normal dist 


    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            values[i][j] = normal(gen); // should convert to a double automatically
            // using he weight initialisation: good for nodes with relu activation -> gaussian with m = 0, sd=sqer(2/inputs)
        }
    }
}

matrix::matrix(int r, int c, double val){
    rows = r;
    columns = c;
    values.resize(rows, vector<double>(columns));
    // note: rxc = num_inputs x numoutputs

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            values[i][j] = val; // should convert to a double automatically
            // using he weight initialisation: good for nodes with relu activation -> gaussian with m = 0, sd=sqer(2/inputs)
        }
    }
}

matrix::matrix(const matrix& other): rows(other.rows), columns(other.columns), values(other.values){}


matrix& matrix::operator=(matrix rhs){
    swap(*this, rhs);
    return *this;
}

matrix::matrix(matrix &&other) noexcept : matrix(){
    swap(*this, other);
}



void swap(matrix& lhs, matrix& rhs){
    using std::swap;
    swap(lhs.rows, rhs.rows);
    swap(lhs.columns, rhs.columns);
    swap(lhs.values, rhs.values);
}

void matrix::clear(){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            values[i][j] = 0.0;
        }
    }
}

std::vector<double>& matrix::operator[] (int idx)
{ 
    if(idx < 0 || idx >= rows){
        throw std::invalid_argument( "out of bounds access attempted on matrix" );
    }
    return values[idx]; 
}


const std::vector<double>& matrix::operator[] (int idx) const
{ 
    if(idx < 0 || idx >= rows){
        throw std::invalid_argument( "out of bounds access attempted on matrix" );
    }

    return values[idx]; 
}


matrix matrix::transpose() const{
    matrix ret(columns, rows, 0);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            ret[j][i]= values[i][j];
        }
    }
    return ret;
}




std::ostream& operator<<(std::ostream& os, const matrix& obj)
{
    // write obj to stream
    for(int i = 0; i < obj.rows; i++){
        for(int j = 0; j < obj.columns; j++){
            os << obj[i][j] << " ";
        }
        os << endl;
    }
    return os;
}

std::istream& operator>>(std::istream& is, matrix& obj){
    for(int i = 0; i < obj.rows; i++){
        for(int j = 0; j < obj.columns; j++){
            if(!(is >> obj[i][j])){
                is.setstate(std::ios::failbit);
            }
        }
    }
    return is;
}

matrix& matrix::operator*=(const double scalar){
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->columns; j++){
            this->values[i][j] = this->values[i][j] * scalar;
        }
    }
    return *this;

}

matrix operator*(const matrix& lhs, const double scalar){
    matrix ret = lhs;
    ret *= scalar;

    return ret;
}

matrix operator*(const matrix &lhs, const matrix& rhs) {

    if(lhs.columns != rhs.rows){
        cout << "matrix\n";
        cout << "lhs " << lhs << endl;
        cout << "rhs " << rhs << endl;
        throw std::invalid_argument( "lhs columns != rhs rows\n" );
    }

    matrix ret(lhs.rows, rhs.columns, 0);
    for(int lhsRow = 0; lhsRow < lhs.rows; lhsRow++){
        for(int rhsColumn = 0; rhsColumn < rhs.columns; rhsColumn++){
            for(int i = 0; i < lhs.columns; i++){
                ret[lhsRow][rhsColumn] += lhs[lhsRow][i] * rhs[i][rhsColumn];
            }
        }
    }

    return ret; // return the result by value (uses move constructor)
}

matrix& matrix::operator*=(const matrix &rhs){
    return *this = *this * rhs;
}

matrix& matrix::operator/=(const double scalar){
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->columns; j++){
            this->values[i][j] = this->values[i][j] / scalar;
        }
    }
    return *this;

}

matrix operator/(const matrix& lhs, const double scalar){
    matrix ret = lhs;
    ret /= scalar;
    return ret;
}


matrix& matrix::operator+=(const matrix&rhs){
    //if right side is too short in terms of rows, will project the size;
    //if left side is too short, returns error
    if(this->columns != rhs.columns){
        cout << this <<  endl << rhs << endl;
        throw std::invalid_argument( "adding column vector to matrix: lhs rows != rhs rows\n" );
    }
    if(this->rows < rhs.rows){
        throw std::invalid_argument( "only the right hand side can be smaller for matrix addition\n" );
    }
    for(int currColumn = 0; currColumn < this->columns; currColumn++){
        for(int currRow = 0; currRow < this->rows; currRow++){
            this->values[currRow][currColumn] = this->values[currRow][currColumn] + rhs[min(currRow, rhs.rows-1)][currColumn]; 
        }
    }

    return *this; // return the result by value (uses move constructor)   
}

matrix operator+(const matrix& lhs, const matrix& rhs) {
    matrix ret = lhs;
    ret += rhs;
    return ret;
}

matrix& matrix::operator-=(const matrix& rhs){
    if(this->columns != rhs.columns){
        throw std::invalid_argument( "adding column vector to matrix: lhs rows != rhs rows\n" );
    }
    if(this->rows < rhs.rows){
        throw std::invalid_argument( "only the right hand side can be smaller for matrix subtraction\n" );
    }
    for(int currColumn = 0; currColumn < this->columns; currColumn++){
        for(int currRow = 0; currRow < this->rows; currRow++){
            this->values[currRow][currColumn] = this->values[currRow][currColumn] - rhs[min(currRow, rhs.rows-1)][currColumn]; 
        }
    }

    return *this; // return the result by value (uses move constructor)   

}

matrix operator-(const matrix &lhs, const matrix& rhs){
    matrix ret = lhs;
    ret -= rhs;
    return ret;
}

bool operator< (const matrix& lhs, const matrix& rhs){

    matrix magnitudeL = lhs * lhs.transpose();
    matrix magnitudeR = rhs * rhs.transpose();

    return magnitudeL < magnitudeR;
}

matrix elementWiseDiv(const matrix& lhs, const matrix& rhs){
    if(lhs.columns != rhs.columns || lhs.rows != rhs.rows){
        throw std::invalid_argument( "your dimensions are incorrect for element wise Div\n" );
    }

    matrix ret(lhs.rows, lhs.columns, 0.0);
    for(int i = 0; i < lhs.rows; i++){
        for(int j = 0; j < lhs.columns; j++){
            ret[i][j] = lhs[i][j] / rhs[i][j];
        }
    }

    return ret;

}

matrix elementWiseMult(const matrix& lhs, const matrix& rhs){
    if(lhs.columns != rhs.columns || lhs.rows != rhs.rows){
        throw std::invalid_argument( "your dimensions are incorrect for element wise Div\n" );
    }

    matrix ret(lhs.rows, lhs.columns, 0.0);
    for(int i = 0; i < lhs.rows; i++){
        for(int j = 0; j < lhs.columns; j++){
            ret[i][j] = lhs[i][j] * rhs[i][j];
        }
    }

    return ret;

}

matrix sqrt(const matrix& lhs){
    matrix ret(lhs.rows, lhs.columns, 0.0);
    for(int i = 0; i < lhs.rows; i++){
        for(int j = 0; j < lhs.columns; j++){
            ret[i][j] = sqrt(lhs[i][j]);
        }
    }

    return ret;

}

double matrix::normalise(){
    if(columns > 1 && rows > 1){
        throw std::invalid_argument( "normalise not implemented for matrices!" );
    }
    else if(rows == 1){
        double result = 0.0;
        for(int i = 0; i < columns; i++){
            result += values[0][i] * values[0][i];
        }
        return result;
    }
    else{
        throw std::invalid_argument( "normalise not implemented for column vectors!" );
    }
}
#include "SVM.h"
#include "matrix.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>

SVM::SVM(std::vector<matrix> & x, std::vector<matrix> & y, double C,  double tol, Kernel k, double sd){
    samples = x;
    labels = y;
    slackC = C;
    // bias = 0;
    tolerance = tol;
    threshold = 0.0; // check if this is correct initialisation

    if(!samples.size() || samples.size() != labels.size()) throw std::invalid_argument( "0 elements, or x != y\n");

    numFeatures = samples[0].columns;
    hyperplane = matrix(numFeatures, 1, 0.0); // only used for the linear kernel
    lagrangianMultipliers.resize(samples.size(), 0.0); 
    errors.resize(samples.size(), 0.0);
    for(int i = 0; i < labels.size(); i++){
        errors[i] = -labels[i][0][0];
    }

    gamma = 1.0/double(numFeatures * sd * sd);
    srand(7);

    // defining the kernel
    kern = k;
    if(k == Kernel::LINEAR){
        computeKernel = [](matrix &a, matrix &b){ 
            // std::cerr <<"attempted kernel" << std::endl;
            return (a*b.transpose())[0][0]; 

        };
    }
    else if(k == Kernel::RBF){
        //implement this;
        //assuming number of dimensions in x1 & x2 are 1.
        //assuming gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of
        // ^ from scikitlearn documentation. Defined above.

        computeKernel = [gamma = this->gamma](matrix &a, matrix&b){ 
            matrix diff = a-b;
            diff = diff*diff.transpose();
            return exp(-(diff[0][0]) * gamma) ;

            //return std::exp(-((a-b).normalise() /2)); ridiculous to have a normalise funciton...
        };
        
    }
    else{
        throw std::invalid_argument( "this is not an available kernel\n");
    }
}

void SVM::SMO(){
    //binary classifier 

    //implementing outer loop of sequential minimal optimisation
    //selects a2 (first of two lagrangians to be optimised)
    int numChanged = 0; // same variable names used in SMO description
    bool examineAll = 1;
    int iteration = 0;
    while((numChanged > 0 || examineAll) && (iteration < 10000)){
        numChanged = 0;
        if(examineAll){
            for(int i = 0; i < samples.size(); i++){
                numChanged += examineSample(i);
            }
        }
        else{
            for(int i = 0; i < samples.size(); i++){
                if(lagrangianMultipliers[i] != 0.0 && lagrangianMultipliers[i] != slackC){
                    numChanged += examineSample(i);
                }
            }
        }

        if(examineAll == 1){
            examineAll = 0;
        }
        else if (numChanged == 0){
            examineAll = 1;
        }

        iteration++;
    }

    
}

bool SVM::examineSample(int i2){
    // std::cerr << "examining Sample" << std::endl;
    //start here
    double y2 = labels[i2].values[0][0];
    double alph2 = lagrangianMultipliers[i2];
    double Error2 = errors[i2]; 
    // if(kern == Kernel::LINEAR){
    //     Error2 = (hyperplane * samples[i2]).values[0][0] - bias - y2;
    // }
    // else if(kern == Kernel::RBF){
    //     //implement this;
    // }
    //check how x is defined rows x columns
    //this is janky accessing of matrix components -> should really define returnValue

    double r2 = Error2*y2;
    if((r2 < -tolerance && alph2 < slackC) || (r2 > tolerance && alph2 > 0)){
        //finish this
        int count = 0;
        for(int i = 0; i < lagrangianMultipliers.size(); i++){
            if(lagrangianMultipliers[i] > 0.0 && lagrangianMultipliers[i] < slackC) count++;
            if(count > 1) break;
        }

        if(count > 1){
            int i1 = 0;
            // if(Error2 > 0){
            //     for(int i = 0; i < errors.size(); i++){
            //         if(errors[i] < errors[i1]) i1 = i;
            //     }
            // }
            // else{
            //     for(int i = 0; i < errors.size(); i++){
            //         if(errors[i] > errors[i1]) i1 = i;
            //     }
            // }

            double maxDiff = -1.0;
            for(int i = 0; i < errors.size(); i++){
                if(i == i2) continue;
                double diff = std::fabs(errors[i] - Error2);
                if(diff > maxDiff){
                    maxDiff = diff;
                    i1 = i;
                }
            }

            if(takeStep(i1, i2)){
                return 1;
            }
        }
        
        // non-zero & non-C lagrangian matrixes.
        for(int i = 0; i < lagrangianMultipliers.size(); i++){
            if(lagrangianMultipliers[i] > 0.0 && lagrangianMultipliers[i] < slackC){
                if(takeStep(i, i2)){
                    return 1;
                }
            }
        }

        //looping over i from random point;
        int randpoint = std::rand() % lagrangianMultipliers.size();
        for(int i = randpoint; i < lagrangianMultipliers.size(); i++){
            if(takeStep(i, i2)){
                return 1;
            }
        }
        for(int i = 0; i < randpoint; i++){
            if(takeStep(i, i2)){
                return 1;
            }
        }
    } 

    return 0;
}

bool SVM::takeStep(int i1, int i2){
    // std::cerr << "taking step" << std::endl;

    if(i1 == i2) return 0;
    double alph1 = lagrangianMultipliers[i1];
    double alph2 = lagrangianMultipliers[i2];
    double y1 = labels[i1].values[0][0];
    double y2 = labels[i2].values[0][0];
    double Error1, Error2; 
    Error1 = 0; Error2 = 0;
    
    // std::cerr << "before kernelling" << std::endl;
    // if(kern == Kernel::LINEAR){
    //     // use error cache or compute fx?
    //     Error1 = (samples[i1] * hyperplane).values[0][0] - threshold - y1;
    //     Error2 = (samples[i2] * hyperplane).values[0][0] - threshold - y2; 
    // }
    // else if(kern == Kernel::RBF){

    //     for(int i = 0; i < samples.size(); i++){
    //         Error1 += labels[i].values[0][0]*lagrangianMultipliers[i]*(computeKernel(samples[i], samples[i1]));
    //         Error2 += labels[i].values[0][0]*lagrangianMultipliers[i]*(computeKernel(samples[i], samples[i2]));

    //     }

    //     Error1 = Error1 - threshold; 
    //     Error2 = Error2 - threshold;
    // }
    Error1 = errors[i1];
    Error2 = errors[i2];

    // std::cerr << "before limits" << std::endl;

    double s = y1*y2; double L, H;
    if(y1 != y2){
        L = std::max(0.0, alph2-alph1);
        H = std::min(slackC, slackC + alph2 - alph1);
    }
    else{
        L = std::max(0.0, alph2 + alph1 - slackC);
        H = std::min(slackC, alph2+alph1);
    }
    

    // std::cerr << "before 2nd kernel" << std::endl;

    if(L == H) return 0;
    double k11 = computeKernel(samples[i1], samples[i1]);
    double k12 = computeKernel(samples[i1], samples[i2]);
    double k22 = computeKernel(samples[i2], samples[i2]);

    // std::cerr << "after 2nd kernel" << std::endl;


    double eta = k11 + k22 - 2.0*k12; // second derivative of objective fx
    double a2;
    if(eta > 0.0){
        a2 = alph2 + y2*(Error1 - Error2) / eta;
        if(a2 < L) a2 = L;
        else if(a2 > H) a2 = H;
    }
    else{
        //need to evaluate objective function
        double f1 = y1*(Error1 + threshold) - alph1*k11 - s*alph2*k12;
        double f2 = y2*(Error1 + threshold) - s*alph1*k12 - alph2*k22;
        double L1 = alph1 + s*(alph2 - L);
        double H1 = alph1 + s*(alph2 - H);
        double objL = L1*f1 + L*f2 + 0.5*L1*L1*k11 + 0.5*L*L*k22 + s*L*L1*k12;
        double objH = H1*f1 + H*f2 + 0.5*H1*H1*k11 + 0.5*H*H*k22 + s*H*H1*k12;

        if(objL < objH - 0.001){ // using epsilon of 0.001
            a2 = L;
        }
        else if(objL > objH + 0.001){
            a2 = H;
        }
        else{
            a2 = alph2;
        }
    }

    // std::cerr << "before abs" << std::endl;
    // std::cerr << abs(a2-alph2) << " " << 0.001*(a2+alph2+0.001) << std::endl;

    if(fabs(a2-alph2) < 0.001*(a2+alph2+0.001)){ //again using epsilon of 0.001
        // std::cerr << "0 error" << std::endl;
        return 0;
    }

    double a1 = alph1 + s*(alph2-a2);

    //calculating new threshold
    // std::cerr << "before threshold calc" << std::endl;

    double b1 = Error1 + y1*(a1-alph1)*k11 + y2*(a2 - alph2)*k12 + threshold;
    double b2 = Error2 + y1*(a1-alph1)*k12 + y2*(a2 - alph2)*k22 + threshold;
    double newbias = 0.0;
    if((0.0 < a1) && (a1 < slackC)){
        newbias = b1;
    }
    else if((0.0 < a2) && (a2 < slackC)){
        newbias = b2;
    }
    else{
        newbias = 0.5 * (b1+b2);
    }


    // update error cache if using it.
    // errors[i1] = 0.0;
    // errors[i2] = 0.0;

    // std::cerr << "before error update" << std::endl;

    for(int i = 0; i < lagrangianMultipliers.size(); i++){
        if(0 < lagrangianMultipliers[i] && lagrangianMultipliers[i] < slackC){
            errors[i] = errors[i] + y1*(a1 - alph1)*computeKernel(samples[i1], samples[i]);
            errors[i] = errors[i] + y2*(a2 - alph2)*computeKernel(samples[i2], samples[i]);
            errors[i] = errors[i] + threshold - newbias;
        } 
    }

    threshold = newbias; // actually updating threshold
    

    // std::cerr << "before hyperplane update" << std::endl;

    // now updating weights (only if linear V+SVM)
    if(kern == Kernel::LINEAR){
        // hyperplane = hyperplane + (samples[i1].transpose())*labels[i1]*lagrangianMultipliers[i1];
        // hyperplane = hyperplane + (samples[i2].transpose())*labels[i2]*lagrangianMultipliers[i2];
        // std::cerr<<"hyerplane: " << hyperplane << std::endl;
        // std::cerr << "sample: " << samples[i1] <<  std::endl;

        hyperplane = hyperplane + (samples[i1].transpose()*y1)*(a1-alph1);

        hyperplane = hyperplane + (samples[i2].transpose()*y2)*(a2-alph2);

        //START HERE CHECK MATH ON THIS W/ PAPER
    }

    //update lagrangian multipliers
    lagrangianMultipliers[i1] = a1;
    lagrangianMultipliers[i2] = a2;


    return 1;
}

double SVM::predict(int i1){
    if(kern == Kernel::LINEAR){
        // use error cache or compute fx?
        return (samples[i1] * hyperplane).values[0][0] - threshold;
    }
    else if(kern == Kernel::RBF){
        double result = 0.0;
        for(int i = 0; i < samples.size(); i++){
            result += labels[i].values[0][0]*lagrangianMultipliers[i]*(computeKernel(samples[i], samples[i1]));
        }

        result = result - threshold; 
        return result;
    }
    else{
        throw std::invalid_argument( "predict not implemented for desired kernel" );
        std::cout.flush();
    }
}

double SVM::predict(matrix i1){
    if(kern == Kernel::LINEAR){
        return (i1 * hyperplane).values[0][0] - threshold;
    }
    else if(kern == Kernel::RBF){
        double result = 0.0;
        for(int i = 0; i < samples.size(); i++){
            result += labels[i].values[0][0]*lagrangianMultipliers[i]*(computeKernel(samples[i], i1));
        }

        result = result - threshold; 
        return result;
    }
    else{
        throw std::invalid_argument( "predict not implemented for desired kernel" );
        std::cout.flush();

    }
}

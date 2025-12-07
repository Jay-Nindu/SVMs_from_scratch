#include "SVM.h"
#include<iostream>


double readIn(std::vector<matrix> &inputs, std::vector<matrix> &outputs, int numSize){
    // reading in X
    double count = 0.0;
    matrix mean(1, 2, 0.0);
    matrix m2(1, 2, 0.0);

    for(int i = 0; i < numSize; i++){
        matrix x(1, 2); // check dimensions
        std::cin >> x;
        count = count+1.0;
        matrix delta = x-mean;
        mean += delta/count;
        matrix delta2 = x-mean;
        m2 += elementWiseMult(delta2, delta);
        inputs.push_back(x);
    }

    m2 = sqrt(m2/numSize); 

    //data normalisation
    for(int i = 0; i < numSize; i++){
        inputs[i] = elementWiseDiv(inputs[i]-mean, m2);
    }

    // reading in Y
    for(int i = 0; i < numSize; i++){
        matrix y(1,1);
        std::cin >> y;
        outputs.push_back(y);
    }

    return m2[0][0];
}



int main(){
    int sampleSizeTrain, sampleSizeTest; std::cin >> sampleSizeTrain >> sampleSizeTest;

    std::vector<matrix> trainInputs, trainOutputs, testInputs, testOutputs; // incase I have test ones.
    double sdtrain = readIn(trainInputs, trainOutputs, sampleSizeTrain); 
    double sdtest = readIn(testInputs, testOutputs, sampleSizeTest);

    double minX, minY, maxX, maxY;
    minX = trainInputs[0][0][0];
    maxX = trainInputs[0][0][0];
    minY = trainInputs[0][0][1];
    maxY = trainInputs[0][0][1];
    for(int i = 1; i < sampleSizeTrain; i++){
        minX = std::min(minX, trainInputs[i][0][0]);
        maxX = std::max(maxX, trainInputs[i][0][0]);

        minY = std::min(minY, trainInputs[i][0][1]);
        maxY = std::max(maxY, trainInputs[i][0][1]);
    }

        std::cout << "maxX " << maxX << std::endl;

    SVM myModel(trainInputs, trainOutputs, 1, 0.001, Kernel::RBF, sdtrain);
    myModel.SMO();

    // for(int i = 0; i < sampleSize; i++){
    //     std::cout << myModel.lagrangianMultipliers[i] << " ";
    // }

    int accuracy = 0;
    for(int i = 0; i < sampleSizeTrain; i++){
        if(trainOutputs[i][0][0] > 0.0 && myModel.predict(trainInputs[i]) > 0.0){
            accuracy++;
            // std::cout << 1 << std::endl;
        }
        else if(trainOutputs[i][0][0] < 0.0 && myModel.predict(trainInputs[i]) < 0.0){
            accuracy++;
            // std::cout << 1 << std::endl;
        }
        else{
            // std::cout << 0 << std::endl;
        }

    }
    std::cout << accuracy << " / " << sampleSizeTrain << " = " << (double(accuracy)/double(sampleSizeTrain)*100.00) << "%" << std::endl;


    for(double i = minX; i < maxX; i = i+(maxX-minX)/50){
        for(double j = minY; j < maxY; j = j + (maxY-minY)/50){
            matrix Sample(1, 2, 0.0);
            Sample[0][0] = i;
            Sample[0][1] = j;
            std::cout << (myModel.predict(Sample) > 0 ? 1 : 0)<< ", ";
        }   
        std::cout << std::endl;
    }


}

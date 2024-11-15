#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <omp.h>

struct Cluster {
    std::vector<double> centroidXVect;
    std::vector<double> centroidYVect;
    std::vector<double> wcssVect;
    std::vector<std::vector<double> > clusterXPointsVect;
    std::vector<std::vector<double> > clusterYPointsVect;
    std::vector<int> clusterPointsNumber;
    std::vector<double> clusterXPointSum;
    std::vector<double> clusterYPointSum;

    Cluster(int kNumber) {
        centroidXVect.resize(kNumber);
        centroidYVect.resize(kNumber);
        wcssVect.resize(kNumber, 0.0);
        clusterXPointsVect.resize(kNumber);
        clusterYPointsVect.resize(kNumber);
        clusterPointsNumber.resize(kNumber);
        clusterXPointSum.resize(kNumber);
        clusterYPointSum.resize(kNumber);
    }
};

void flushCache() {
    const size_t cacheSize = 10 * 1024 * 1024;
    char *temp = new char[cacheSize];
    for (size_t i = 0; i < cacheSize; i++) {
        temp[i] = i % 256;
    }
    delete [] temp;
}

int main() {
    flushCache();
    //srand(time(NULL));
    int kNumber = 60;
    int pointsNumber = 100000;
    int maxRand = 1000;

    /*Init xCoordVector and yCoordVector*/
    std::vector<double> pointX(pointsNumber);
    std::vector<double> pointY(pointsNumber);

    /*Assigns random values to every x and y coord (for debug purposes always the same values will be returned by rand() function. Activate srand()!)*/
    for (int i = 0; i < pointsNumber; ++i) {
        pointX[i] = rand() % maxRand;
        pointY[i] = rand() % maxRand;
    }

    /*Init Centroids as the firsts k Points*/
    Cluster clusterStruct(kNumber);
    for (int i = 0; i < kNumber; ++i) {
        clusterStruct.centroidXVect[i] = pointX[i];
        clusterStruct.centroidYVect[i] = pointY[i];
    }
    bool maxConvergenceAchieved = false;
    bool lastIteration = false;
    int iterations = 0;
    int actMinDistanceIndex = 0;
    double actMinDistance = 0.0;
    double previousWCSS = 0.0;
    double actualWCSS = 0.0;
    auto beginTime = std::chrono::high_resolution_clock::now();
    while (!maxConvergenceAchieved || !lastIteration) {
        if (maxConvergenceAchieved) {
            lastIteration = true;
        }
        maxConvergenceAchieved = true;
        iterations++;
        /*Deletes Points and wcss from Clusters*/
        for (int i = 0; i < kNumber; ++i) {
            clusterStruct.clusterXPointsVect[i].clear();
            clusterStruct.clusterYPointsVect[i].clear();
            clusterStruct.clusterPointsNumber[i] = 0;
            clusterStruct.clusterXPointSum[i] = 0.0;
            clusterStruct.clusterYPointSum[i] = 0.0;
            clusterStruct.wcssVect[i] = 0.0;
        }
        /*Calculates the distance between every Point and the Clusters; assigns the Point to the nearest one.*/
        for (int i = kNumber; i < pointsNumber; ++i) {
            /*Assigns arbitrary the minimum distance to the 0-indexed Cluster*/
            actMinDistanceIndex = 0;
            actMinDistance =
                    (clusterStruct.centroidXVect[0] - pointX[i]) * (
                        clusterStruct.centroidXVect[0] - pointX[i]) + (
                        clusterStruct.centroidYVect[0] - pointY[i]) * (
                        clusterStruct.centroidYVect[0] - pointY[i]);
            for (int j = 1; j < kNumber; ++j) {
                double actDistance =
                        (clusterStruct.centroidXVect[j] - pointX[i]) * (
                            clusterStruct.centroidXVect[j] - pointX[i]) + (
                            clusterStruct.centroidYVect[j] - pointY[i]) * (
                            clusterStruct.centroidYVect[j] - pointY[i]);
                if (actDistance < actMinDistance) {
                    actMinDistance = actDistance;
                    actMinDistanceIndex = j;
                }
            }
            if (lastIteration) {
                clusterStruct.clusterXPointsVect[actMinDistanceIndex].push_back(pointX[i]);
                clusterStruct.clusterYPointsVect[actMinDistanceIndex].push_back(pointY[i]);
            }
            clusterStruct.clusterPointsNumber[actMinDistanceIndex] += 1;
            clusterStruct.clusterXPointSum[actMinDistanceIndex] += pointX[i];
            clusterStruct.clusterYPointSum[actMinDistanceIndex] += pointY[i];
            actualWCSS += actMinDistance * actMinDistance;
        }
        if (!lastIteration) {
            double acceptedDelta = 0.001;
            if (iterations == 1) {
                previousWCSS = actualWCSS;
                actualWCSS = 0.0;
                maxConvergenceAchieved = false;
            } else {
                double actDelta = previousWCSS - actualWCSS;
                previousWCSS = actualWCSS;
                actualWCSS = 0.0;
                maxConvergenceAchieved = actDelta <= acceptedDelta;
            }
        }
        /*END OF PARALLEL REGION*/
        if (!maxConvergenceAchieved) {
            double xSum = 0.0;
            double ySum = 0.0;
            int totalClusterPoints = 0;
            int i = 0;
            for (i = 0; i < kNumber; ++i) {
                xSum = clusterStruct.clusterXPointSum[i];
                ySum = clusterStruct.clusterYPointSum[i];
                totalClusterPoints = clusterStruct.clusterPointsNumber[i];
                double xBar = xSum / totalClusterPoints;
                double yBar = ySum / totalClusterPoints;
                clusterStruct.centroidXVect[i] = xBar;
                clusterStruct.centroidYVect[i] = yBar;
            }
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = endTime - beginTime;
    printf("Duration ms: %f\n", ms_double.count());
    printf("IT:%d\n", iterations - 1);
    return 0;
}

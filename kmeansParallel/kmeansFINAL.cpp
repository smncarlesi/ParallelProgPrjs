#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <omp.h>

struct Cluster {
    std::vector<double> centroidXVect;
    std::vector<double> centroidYVect;
    std::vector<int> clusterPointsNumber;
    std::vector<double> clusterXPointSum;
    std::vector<double> clusterYPointSum;
    std::vector<double> wcssVect;
    std::vector<std::vector<double> > clusterXPointsVect;
    std::vector<std::vector<double> > clusterYPointsVect;

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
#pragma omp parallel default(none)
    {
        flushCache();
    }
    //srand(time(NULL));
    int kNumber = 3;
    int pointsNumber = 100000000;
    int maxRand = 1000;

    /*Init xCoordVector and yCoordVector*/
    std::vector<double> pointX(pointsNumber);
    std::vector<double> pointY(pointsNumber);

    /*Assigns pseudo-pseudo-random values to every x and y coord*/
    for (int i = 0; i < pointsNumber; ++i) {
        pointX[i] = rand() % maxRand;
        pointY[i] = rand() % maxRand;
    }

    /*Init Centroids as the firsts k Points*/
    Cluster clusterStruct(kNumber);
    for (int i = 0; i < kNumber; ++i) {
        double act_PointX = pointX[i];
        double act_PointY = pointY[i];
        clusterStruct.centroidXVect[i] = act_PointX;
        clusterStruct.centroidYVect[i] = act_PointY;
    }
    bool maxConvergenceAchieved = false;
    bool lastIteration = false;
    int iterations = 0;
    int actMinDistanceIndex = 0;
    double actMinDistance = 0.0;
    double previousWCSS = 0.0;
    double actualWCSS = 0.0;
    int myStartIndex = 0;
    int myEndIndex = 0;
    int remainder = 0;
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
#pragma omp parallel default(none) shared(maxConvergenceAchieved, actualWCSS, previousWCSS, pointX, pointY, clusterStruct) firstprivate(lastIteration, iterations, remainder, myStartIndex, myEndIndex, kNumber, pointsNumber, actMinDistanceIndex, actMinDistance) if(pointsNumber > 59000)
        {
            Cluster privateClusterStruct(kNumber);
            /*Makes a local copy of Cluster Struct Centroid coordinates for every thread*/
            for (int i = 0; i < kNumber; ++i) {
                privateClusterStruct.centroidXVect[i] = clusterStruct.centroidXVect[i];
                privateClusterStruct.centroidYVect[i] = clusterStruct.centroidYVect[i];
            }
            int num_threads = omp_get_num_threads();
            int myID = omp_get_thread_num();
            int effectivePointsNumber = pointsNumber - kNumber;
            remainder = effectivePointsNumber % num_threads;
            if (myID != 0) {
                myStartIndex = myID < remainder
                                   ? ((effectivePointsNumber / num_threads) + 1) * myID + kNumber
                                   : ((effectivePointsNumber / num_threads) * myID) + remainder + kNumber;
            } else {
                myStartIndex = kNumber;
            }
            myEndIndex = myID < remainder
                             ? myStartIndex + (effectivePointsNumber / num_threads)
                             : (myStartIndex + (effectivePointsNumber / num_threads)) - 1;
            /*Calculates the distance between every Point and the Clusters; assigns the Point to the nearest one.*/
            for (int i = myStartIndex; i <= myEndIndex; ++i) {
                double act_pointX = pointX[i];
                double act_PointY = pointY[i];
                /*Assigns arbitrary the minimum distance to the 0-indexed Cluster*/
                actMinDistanceIndex = 0;
                actMinDistance =
                        (privateClusterStruct.centroidXVect[0] - act_pointX) * (
                            privateClusterStruct.centroidXVect[0] - act_pointX) + (
                            privateClusterStruct.centroidYVect[0] - act_PointY) * (
                            privateClusterStruct.centroidYVect[0] - act_PointY);
                for (int j = 1; j < kNumber; ++j) {
                    double actDistance =
                            (privateClusterStruct.centroidXVect[j] - act_pointX) * (
                                privateClusterStruct.centroidXVect[j] - act_pointX) + (
                                privateClusterStruct.centroidYVect[j] - act_PointY) * (
                                privateClusterStruct.centroidYVect[j] - act_PointY);
                    if (actDistance < actMinDistance) {
                        actMinDistance = actDistance;
                        actMinDistanceIndex = j;
                    }
                }
                if (lastIteration) {
                    privateClusterStruct.clusterXPointsVect[actMinDistanceIndex].push_back(pointX[i]);
                    privateClusterStruct.clusterYPointsVect[actMinDistanceIndex].push_back(pointY[i]);
                }
                privateClusterStruct.clusterPointsNumber[actMinDistanceIndex] += 1;
                privateClusterStruct.clusterXPointSum[actMinDistanceIndex] += pointX[i];
                privateClusterStruct.clusterYPointSum[actMinDistanceIndex] += pointY[i];
                privateClusterStruct.wcssVect[actMinDistanceIndex] += actMinDistance * actMinDistance;
            }
            /*Updates the global WCSS*/
            if (!lastIteration) {
                for (int i = 0; i < kNumber; ++i) {
#pragma omp atomic
                    actualWCSS += privateClusterStruct.wcssVect[i];
                }
            }
#pragma omp barrier
#pragma omp single nowait
            {
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
            }
            /*Makes a copy of the partial structs into the global one at the end of the process*/
            if (lastIteration) {
#pragma omp critical
                {
                    for (int i = 0; i < kNumber; ++i) {
                        clusterStruct.clusterXPointsVect[i].insert(clusterStruct.clusterXPointsVect[i].end(),
                                                                   privateClusterStruct.clusterXPointsVect[i].
                                                                   begin(),
                                                                   privateClusterStruct.clusterXPointsVect[i].
                                                                   end());
                        clusterStruct.clusterYPointsVect[i].insert(clusterStruct.clusterYPointsVect[i].end(),
                                                                   privateClusterStruct.clusterYPointsVect[i].
                                                                   begin(),
                                                                   privateClusterStruct.clusterYPointsVect[i].
                                                                   end());
                    }
                } /*If convergence not reached, updates the global clusterStruct*/
            } else {
#pragma omp critical
                {
                    for (int i = 0; i < kNumber; ++i) {
                        clusterStruct.clusterXPointSum[i] += privateClusterStruct.clusterXPointSum[i];
                        clusterStruct.clusterYPointSum[i] += privateClusterStruct.clusterYPointSum[i];
                        clusterStruct.clusterPointsNumber[i] += privateClusterStruct.clusterPointsNumber[i];
                    }
                }
            }
        }
        /*END OF PARALLEL REGION*/
        if (!maxConvergenceAchieved) {
            double xSum = 0.0;
            double ySum = 0.0;
            int totalClusterPoints = 0;
#pragma omp parallel for default(none) firstprivate(xSum, ySum, totalClusterPoints) shared(clusterStruct, kNumber) schedule(static) if(kNumber > 59)
            for (int i = 0; i < kNumber; ++i) {
                double xPSum = clusterStruct.clusterXPointSum[i];
                double YPSum = clusterStruct.clusterYPointSum[i];
                xSum = xPSum;
                ySum = YPSum;
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

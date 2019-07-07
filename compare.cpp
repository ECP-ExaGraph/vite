// ***********************************************************************
//
//            Vite: A C++ library for distributed-memory graph clustering 
//                  using MPI+OpenMP
// 
//               Daniel Chavarria (daniel.chavarria@pnnl.gov)
//               Antonino Tumeo (antonino.tumeo@pnnl.gov)
//               Mahantesh Halappanavar (hala@pnnl.gov)
//               Pacific Northwest National Laboratory	
//
//               Hao Lu (luhowardmark@wsu.edu)
//               Sayan Ghosh (sayan.ghosh@wsu.edu)
//               Ananth Kalyanaraman (ananth@eecs.wsu.edu)
//               Washington State University
//
// ***********************************************************************
//
//       Copyright (2017) Battelle Memorial Institute
//                      All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************

#include "compare.hpp"

extern std::ofstream ofs;

//Only a single process performs the cluster comparisons
//Assume that clusters have been numbered in a contiguous manner
//Assume that C1 is the truth data
void compare_communities(std::vector<GraphElem> const& C1, 
        std::vector<GraphElem> const& C2) {

    // number of vertices of two sets must match
    const GraphElem N1 = C1.size();
    const GraphElem N2 = C2.size();

    assert(N1>0 && N2>0);
    assert(N1 == N2);
   
    int nT;
#pragma omp parallel
    {
        nT = omp_get_num_threads();
    }

    //Compute number of communities in each set:
    GraphElem nC1=-1;
#pragma omp parallel for reduction(max: nC1)
    for(GraphElem i = 0; i < N1; i++) {
        if (C1[i] > nC1) {
            nC1 = C1[i];
        }
    }

    nC1++; // nC1 contains index, so need to +1

    GraphElem nC2=-1;
#pragma omp parallel for reduction(max: nC2)
    for(GraphElem i = 0; i < N2; i++) {
        if (C2[i] > nC2) {
            nC2 = C2[i];
        }
    }

    nC2++;

#if defined(DONT_CREATE_DIAG_FILES)
    std::cout << "Number of unique communities in C1 = " << nC1 << 
        ", and C2 = " << nC2 << std::endl;
#else
    ofs << "Number of unique communities in C1 = " << nC1 << 
        ", and C2 = " << nC2 << std::endl;
#endif

    //////////STEP 1: Create a CSR-like datastructure for communities in C1
    GraphElem* commPtr1 = new GraphElem[nC1+1];
    GraphElem* commIndex1 = new GraphElem[N1];
    GraphElem* commAdded1 = new GraphElem[nC1];
    GraphElem* clusterDist1 = new GraphElem[nC1]; //For Gini coefficient
    
    // Initialization
#pragma omp parallel for
    for(GraphElem i = 0; i < nC1; i++) {
        commPtr1[i] = 0;
        commAdded1[i] = 0;
    }
    commPtr1[nC1] = 0;
    
    // Count the size of each community
#pragma omp parallel for
    for(GraphElem i = 0; i < N1; i++) {
#pragma omp atomic update
            commPtr1[C1[i]+1]++;
    }

#pragma omp parallel for
    for(GraphElem i = 0; i < nC1; i++) {
        clusterDist1[i] = commPtr1[i+1]; //Zeroth position is not valid
    }
    
    //Prefix sum:
    for(GraphElem i=0; i<nC1; i++) {
        commPtr1[i+1] += commPtr1[i];
    }
    
    //Group vertices with the same community in particular order
#pragma omp parallel for
    for (GraphElem i=0; i<N1; i++) {
        long ca;
#pragma omp atomic capture
        ca = commAdded1[C1[i]]++;
        GraphElem Where = commPtr1[C1[i]] + ca;
        commIndex1[Where] = i; //The vertex id
    }

    delete []commAdded1;
#ifdef DEBUG_PRINTF     
    ofs << "Done building structure for C1..." << std::endl;
#endif

    //////////STEP 2: Create a CSR-like datastructure for communities in C2
    GraphElem* commPtr2 = new GraphElem[nC2+1];
    GraphElem* commIndex2 = new GraphElem[N2];
    GraphElem* commAdded2 = new GraphElem[nC2];
    GraphElem* clusterDist2 = new GraphElem[nC2]; //For Gini coefficient   
   
    // Initialization
#pragma omp parallel for
    for(GraphElem i = 0; i < nC2; i++) {
        commPtr2[i] = 0;
        commAdded2[i] = 0;
    }
    commPtr2[nC2] = 0;
    
    // Count the size of each community
#pragma omp parallel for
    for(GraphElem i = 0; i < N2; i++) {
#pragma omp atomic update
            commPtr2[C2[i]+1]++;
    }

#pragma omp parallel for
    for(GraphElem i = 0; i < nC2; i++) {
        clusterDist2[i] = commPtr2[i+1]; //Zeroth position is not valid
    }
    
    //Prefix sum:
    for(GraphElem i=0; i<nC2; i++) {
        commPtr2[i+1] += commPtr2[i];
    }
    
    //Group vertices with the same community in particular order
#pragma omp parallel for
    for (GraphElem i=0; i<N2; i++) {
        long ca;
#pragma omp atomic capture
        ca = commAdded2[C2[i]]++;
        GraphElem Where = commPtr2[C2[i]] + ca;
        commIndex2[Where] = i; //The vertex id
    }

    delete []commAdded2;
#ifdef DEBUG_PRINTF     
    ofs << "Done building structure for C2..." << std::endl;
#endif   
    
    //////////STEP 3:  Compute statistics:
    // thread-private vars
    GraphElem tSameSame[nT], tSameDiff[nT], tDiffSame[nT], nAgree[nT];
#pragma omp parallel for
    for (int i=0; i < nT; i++) {
        tSameSame[i] = 0;
        tSameDiff[i] = 0;
        tDiffSame[i] = 0;
        nAgree[i]    = 0;
    }

    //Compare all pairs of vertices from the perspective of C1 (ground truth):
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for
        for(GraphElem ci = 0; ci < nC1; ci++) {
            GraphElem adj1 = commPtr1[ci];
            GraphElem adj2 = commPtr1[ci+1];
            for(GraphElem i = adj1; i < adj2; i++) {
                for(GraphElem j = i+1; j < adj2; j++) {
                    //Check if the two vertices belong to the same community in C2
                    if(C2[commIndex1[i]] == C2[commIndex1[j]]) {
                        tSameSame[tid]++;
                    } else {
                        tSameDiff[tid]++;
                    }
                }//End of for(j)
            }//End of for(i)
        }//End of for(ci)
    }//End of parallel region
#ifdef DEBUG_PRINTF     
    ofs << "Done parsing C1..." << std::endl;
#endif

    //Compare all pairs of vertices from the perspective of C2:
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for
        for(GraphElem ci = 0; ci < nC2; ci++) {
            GraphElem adj1 = commPtr2[ci];
            GraphElem adj2 = commPtr2[ci+1];
            for(GraphElem i = adj1; i < adj2; i++) {
                for(GraphElem j = i+1; j < adj2; j++) {
                    //Check if the two vertices belong to the same community in C2
                    if(C1[commIndex2[i]] == C1[commIndex2[j]]) {
                        nAgree[tid]++;
                    } else {
                        tDiffSame[tid]++;
                    }
                }//End of for(j)
            }//End of for(i)
        }//End of for(ci)
    }//End of parallel region

#ifdef DEBUG_PRINTF     
    ofs << "Done parsing C2..." << std::endl;
#endif
    
    GraphElem SameSame = 0, SameDiff = 0, DiffSame = 0, Agree = 0;
#pragma omp parallel for reduction(+:SameSame) reduction(+:SameDiff) \
reduction(+:DiffSame) reduction(+:Agree)
    for (int i=0; i < nT; i++) {
        SameSame += tSameSame[i];
        SameDiff += tSameDiff[i];
        DiffSame += tDiffSame[i];
        Agree    += nAgree[i];
    }

    assert(SameSame == Agree);
    
    double precision = (double)SameSame / (double)(SameSame + DiffSame);
    double recall    = (double)SameSame / (double)(SameSame + SameDiff);
    
    //F-score (F1 score) is the harmonic mean of precision and recall --
    //multiplying the constant of 2 scales the score to 1 when both recall and precision are 1
    double fScore = 2.0*((precision * recall) / (precision + recall));
    
    //Compute Gini coefficient for each cluster:
    double Gini1 = compute_gini_coeff(clusterDist1, nC1);
    double Gini2 = compute_gini_coeff(clusterDist2, nC2);

    std::cout << "*******************************************" << std::endl;
    std::cout << "Communities comparison statistics:" << std::endl;
    std::cout << "*******************************************" << std::endl;
    std::cout << "|C1| (truth)       : " << N1 << std::endl;
    std::cout << "#communities in C1 : " << nC1 << std::endl;
    std::cout << "|C2| (output)      : " << N2 << std::endl;
    std::cout << "#communities in C2 : " << nC2 << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Same-Same (True positive)  : " << SameSame << std::endl;
    std::cout << "Same-Diff (False negative) : " << SameDiff << std::endl;
    std::cout << "Diff-Same (False positive) : " << DiffSame << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Precision :  " << precision << " (" << (precision*100) << ")" << std::endl;
    std::cout << "Recall    :  " << recall << " (" << (recall*100) << ")" << std::endl;
    std::cout << "F-score   :  " << fScore << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Gini coefficient, C1  :  " << Gini1 << std::endl;
    std::cout << "Gini coefficient, C2  :  " << Gini2 << std::endl;
    std::cout << "*******************************************" << std::endl;
    
    //Cleanup:
    delete []commPtr1; 
    delete []commIndex1;
    delete []commPtr2; 
    delete []commIndex2;
    delete []clusterDist1; 
    delete []clusterDist2;
    
} //End of compare_communities(...)

//WARNING: Assume that colorSize is populated with the frequency for each color
//Will sort the array within the function
double compute_gini_coeff(GraphElem *colorSize, int numColors) {
    //Step 1: Sort the color size vector -- use STL sort function
#ifdef DEBUG_PRINTF     
    double time1 = omp_get_wtime();
#endif
    std::sort(colorSize, colorSize+numColors);
#ifdef DEBUG_PRINTF     
    double time2 = omp_get_wtime();
    ofs << "Time for sorting (in compute_gini_coeff): " 
        << time2-time1 << std::endl;
#endif
    //Step 2: Compute Gini coefficient
    double numFunc=0.0, denFunc=0.0;
#pragma omp parallel for reduction(+:numFunc,denFunc)
    for (GraphElem i=0; i < numColors; i++) {
        numFunc = numFunc + ((i+1)*colorSize[i]);
        denFunc = denFunc + colorSize[i];
    }
#ifdef DEBUG_PRINTF     
    ofs << "(Gini coeff) Numerator = " << numFunc << " Denominator = " << denFunc << std::endl;
    ofs << "(Gini coeff) Negative component = " << ((double)(numColors+1)/(double)numColors) << std::endl;
#endif

    double giniCoeff = ((2*numFunc)/(numColors*denFunc)) - ((double)(numColors+1)/(double)numColors);
    
    return giniCoeff; //Return the Gini coefficient
}//End of compute_gini_coeff(...)


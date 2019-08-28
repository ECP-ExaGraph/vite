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

#ifndef __UTILS_H
#define __UTILS_H

#include "graph.hpp"

#include <cstring>
#include <random>
#include <utility>

#ifndef RANDOM_MAX_WEIGHT
#define RANDOM_MAX_WEIGHT (1.0)
#endif
#ifndef RANDOM_MIN_WEIGHT
#define RANDOM_MIN_WEIGHT (0.01)
#endif

#ifndef TERMINATION_PHASE_COUNT
#define TERMINATION_PHASE_COUNT (200)
#endif

// Read https://en.wikipedia.org/wiki/Linear_congruential_generator#Period_length
// about choice of LCG parameters
// From numerical recipes
// TODO FIXME investigate larger periods
#define MLCG                        (2147483647)    // 2^31 - 1
#define ALCG                        (16807)         // 7^5
#define BLCG                        (0)
#define SR_LCG_TAG                  108

struct GraphElemTuple {

    GraphElem i_, j_;
    GraphWeight w_;

    GraphElemTuple(GraphElem i, GraphElem j, GraphWeight w): 
        i_(i), j_(j), w_(w) 
    {}
    GraphElemTuple(GraphElem i, GraphElem j): 
        i_(i), j_(j), w_(1.0) 
    {}
    GraphElemTuple(): 
        i_(-1), j_(-1), w_(0.0) 
    {}

    // compare   
    bool operator <(GraphElemTuple const& tp) const
    { return (i_ < tp.i_) || ((!(tp.i_ < i_)) && (j_ < tp.j_)); }
};

typedef enum 
{
    RND_WEIGHT,    // random real weight, between 0-1
    ONE_WEIGHT,    // weight = 1
    ORG_WEIGHT,    // use original weights of graph
    ABS_WEIGHT     // use absolute original weights of graph
} Weight_t;

extern unsigned seed;

double mytimer(void);

// uses a static random engine (seed)
GraphWeight genRandom(GraphWeight low, GraphWeight high);

// sort edge list
void processGraphData(Graph &g, std::vector<GraphElem> &edgeCount,
		      std::vector<GraphElemTuple> &edgeList,
		      const GraphElem nv, const GraphElem ne);    

// forward declare LCG class for 
// distributed random number 
// generation

// Parallel Linear Congruential Generator
// x[i] = (a*x[i-1] + b)%M
class LCG
{
    public:
        LCG(unsigned seed, GraphWeight* drand, 
            GraphElem n, MPI_Comm comm = MPI_COMM_WORLD): 
        seed_(seed), drand_(drand), n_(n)
        {
            comm_ = comm;
            MPI_Comm_size(comm_, &nprocs_);
            MPI_Comm_rank(comm_, &rank_);

            // allocate long random numbers
            rnums_.resize(n_);

            // init x0
            if (rank_ == 0)
                x0_ = reseeder(seed_);

            // step #1: bcast x0 from root
            MPI_Bcast(&x0_, 1, MPI_GRAPH_TYPE, 0, comm_);
            
            // step #2: parallel prefix to generate first random value per process
            parallel_prefix_op();
        }
        
        ~LCG() { rnums_.clear(); }

        // return unint32_t seed
        GraphElem reseeder(unsigned initseed)
        {
            std::seed_seq seq({initseed});
            std::vector<std::uint32_t> seeds(1);
            seq.generate(seeds.begin(), seeds.end());

            return (GraphElem)seeds[0];
        }

        // matrix-matrix multiplication for 2x2 matrices
        void matmat_2x2(GraphElem c[], GraphElem a[], GraphElem b[])
        {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    GraphElem sum = 0;
                    for (int k = 0; k < 2; k++) {
                        sum += a[i*2+k]*b[k*2+j];
                    }
                    c[i*2+j] = sum;
                }
            }
        }

        // x *= y
        void matop_2x2(GraphElem x[], GraphElem y[])
        {
            GraphElem tmp[4];
            matmat_2x2(tmp, x, y);
            memcpy(x, tmp, sizeof(GraphElem[4]));
        }

        // find kth power of a 2x2 matrix
        void mat_power(GraphElem mat[], GraphElem k)
        {
            GraphElem tmp[4];
            memcpy(tmp, mat, sizeof(GraphElem[4]));

            // mat-mat multiply k times
            for (GraphElem p = 0; p < k-1; p++)
                matop_2x2(mat, tmp);
        }

        // parallel prefix for matrix-matrix operation
        // `x0 is the very first random number in the series
        // `ab is a 2-length array which stores a and b
        // `n_ is (n/p)
        // `rnums is n_ length array which stores the random nums for a process
        void parallel_prefix_op()
        {
            GraphElem global_op[4]; 
            global_op[0] = ALCG;
            global_op[1] = 0;
            global_op[2] = BLCG;
            global_op[3] = 1;

            mat_power(global_op, n_);        // M^(n/p)
            GraphElem prefix_op[4] = {1,0,0,1};  // I in row-major

            GraphElem global_op_recv[4];

            int steps = (int)(log2((double)nprocs_));

            for (int s = 0; s < steps; s++) {
                
                int mate = rank_^(1 << s); // toggle the sth LSB to find my neighbor
                
                // send/recv global to/from mate
                MPI_Sendrecv(global_op, 4, MPI_GRAPH_TYPE, mate, SR_LCG_TAG, 
                        global_op_recv, 4, MPI_GRAPH_TYPE, mate, SR_LCG_TAG, 
                        comm_, MPI_STATUS_IGNORE);

                matop_2x2(global_op, global_op_recv);   
                
                if (mate < rank_) 
                    matop_2x2(prefix_op, global_op_recv);

                MPI_Barrier(comm_);
            }

            // populate the first random number entry for each process
            // (x0*a + b)%P
            if (rank_ == 0)
                rnums_[0] = x0_;
            else
                rnums_[0] = (x0_*prefix_op[0] + prefix_op[2])%MLCG;
        }

        // generate random number based on the first 
        // random number on a process
        // TODO check the 'quick'n dirty generators to
        // see if we can avoid the mod
        void generate()
        {
#if defined(PRINT_LCG_LONG_RANDOM_NUMBERS)
            for (int k = 0; k < nprocs_; k++) {
                if (k == rank_) {
                    std::cout << "------------" << std::endl;
                    std::cout << "Process#" << rank_ << " :" << std::endl;
                    std::cout << "------------" << std::endl;
                    std::cout << rnums_[0] << std::endl;
                    for (GraphElem i = 1; i < n_; i++) {
                        rnums_[i] = (rnums_[i-1]*ALCG + BLCG)%MLCG;
                        std::cout << rnums_[i] << std::endl;
                    }
                }
                MPI_Barrier(comm_);
            }
#else
            for (GraphElem i = 1; i < n_; i++) {
                rnums_[i] = (rnums_[i-1]*ALCG + BLCG)%MLCG;
            }
#endif
            GraphWeight mult = 1.0 / (GraphWeight)(1.0 + (GraphWeight)(MLCG-1));

#if defined(PRINT_LCG_DOUBLE_RANDOM_NUMBERS)
            for (int k = 0; k < nprocs_; k++) {
                if (k == rank_) {
                    std::cout << "------------" << std::endl;
                    std::cout << "Process#" << rank_ << " :" << std::endl;
                    std::cout << "------------" << std::endl;

                    for (GraphElem i = 0; i < n_; i++) {
                        drand_[i] = (GraphWeight)((GraphWeight)fabs(rnums_[i]) * mult ); // 0-1
                        std::cout << drand_[i] << std::endl;
                    }
                }
                MPI_Barrier(comm_);
            }
#else
            for (GraphElem i = 0; i < n_; i++)
                drand_[i] = (GraphWeight)((GraphWeight)fabs(rnums_[i]) * mult); // 0-1
#endif
        }
         
        // copy from drand_[idx_start] to new_drand, 
        // rescale the random numbers between lo and hi
        void rescale(GraphWeight* new_drand, GraphElem idx_start, GraphWeight const& lo)
        {
            GraphWeight range = (1.0 / (GraphWeight)nprocs_);

#if defined(PRINT_LCG_DOUBLE_LOHI_RANDOM_NUMBERS)
            for (int k = 0; k < nprocs_; k++) {
                if (k == rank_) {
                    std::cout << "------------" << std::endl;
                    std::cout << "Process#" << rank_ << " :" << std::endl;
                    std::cout << "------------" << std::endl;

                    for (GraphElem i = idx_start, j = 0; i < n_; i++, j++) {
                        new_drand[j] = lo + (GraphWeight)(range * drand_[i]);
                        std::cout << new_drand[j] << std::endl;
                    }
                }
                MPI_Barrier(comm_);
            }
#else
            for (GraphElem i = idx_start, j = 0; i < n_; i++, j++)
                new_drand[j] = lo + (GraphWeight)(range * drand_[i]); // lo-hi
#endif
        }

    private:
        MPI_Comm comm_;
        int nprocs_, rank_;
        unsigned seed_;
        GraphElem n_, x0_;
        GraphWeight* drand_;
        std::vector<GraphElem> rnums_;
};

#endif // __UTILS_H

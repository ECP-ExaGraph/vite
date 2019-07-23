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

double mytimer(void);

// uses a static random engine (seed)
GraphWeight genRandom(GraphWeight low, GraphWeight high);

// sort edge list
void processGraphData(Graph &g, std::vector<GraphElem> &edgeCount,
		      std::vector<GraphElemTuple> &edgeList,
		      const GraphElem nv, const GraphElem ne);     
#endif // __UTILS_H

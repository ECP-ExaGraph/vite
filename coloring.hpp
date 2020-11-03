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
#ifndef __COLORING_H
#define __COLORING_H

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <numeric>
#include <vector>
#include <unordered_set>
#include <map>

#include <mpi.h>
#include <omp.h>

#include "graph.hpp"
#include "distgraph.hpp"

#ifndef MAX_COVG
#define MAX_COVG    (70)
#endif


#ifdef USE_32_BIT_GRAPH
typedef int32_t ColorElem;
#else
typedef int64_t ColorElem;
#endif

const int ColoringSizeTag = 6;
const int ColoringDataTag = 7;

typedef std::unordered_set<GraphElem> ColoredVertexSet; 
typedef std::vector<ColorElem> ColorVector;

ColorElem distColoringMultiHashMinMax(const int me, const int nprocs, const DistGraph &dg, ColorVector &vertexColor, const ColorElem nHash, const int target_percent, const bool singleIteration);

static unsigned int hash(unsigned int a, unsigned int seed);

void distColoringIteration(const int me, const DistGraph &dg, ColorVector &vertexColor, ColoredVertexSet &remoteColoredVertices, const ColorElem nHash, const ColorElem nextColor, const unsigned int seed);

void setUpGhostVertices(const int me, const int nprocs, const DistGraph &dg, std::vector<GraphElem> &ghostVertices, std::vector<GraphElem> &ghostSizes);

void sendColoredRemoteVertices(const int me, const int nprocs, const DistGraph &dg, ColoredVertexSet &remoteColoredVertices, const ColorVector &vertexColor, const std::vector<GraphElem> &ghostVertices, const std::vector<GraphElem> &ghostSizes);

GraphElem countUnassigned(const ColorVector &vertexColor);

GraphElem distCheckColoring(const int me, const int nprocs, const DistGraph &dg, const ColorVector &vertexColor, const std::vector<GraphElem> &ghostVertices, const std::vector<GraphElem> &ghostSizes);	

#endif 

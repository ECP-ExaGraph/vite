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
#include <cmath>
#include <omp.h>
#include <random>
#include <functional>
#include <iostream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "nkit-er.hpp"
#include "../utils.hpp"

#include <NetworKit/Globals.h>

#include <NetworKit/graph/GraphBuilder.h>

void generateER(Graph *&g, unsigned int N, double p, bool randomEdgeWeight)
{
    if (!N || p <= 0.0)
    {
        std::cerr << "For ER graph, N and p must be specified!" << std::endl;
        exit(EXIT_FAILURE);
    }

    NetworKit::Graph G;
    NetworKit::GraphBuilder builder;

    // prepare a random generator for each possible thread
    int maxThreads = omp_get_max_threads();
    std::vector< std::function<double()> > randomPerThread;
    std::random_device device;
    std::uniform_int_distribution<uint64_t> intDist;
    for (int tid = 0; tid < maxThreads; tid++) {
        auto seed = intDist(device);
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<double> dist{0.0, std::nexttoward(1.0, 2.0)};
        auto rdn = std::bind(dist, gen);
        randomPerThread.push_back(rdn);
    }

    double t1, t2;

    t1 = mytimer(); 
    builder = NetworKit::GraphBuilder(N);
    builder.parallelForNodePairs([&](NetworKit::node u, NetworKit::node v) {
            int tid = omp_get_thread_num();
            double rdn = randomPerThread[tid]();
            if (rdn <= p) {
            builder.addHalfEdge(u, v);
            }
            });
    G = builder.toGraph(true, false);

    /* Generate binary file containing edgelist */
    GraphElem numEdges = G.numberOfEdges();
    GraphElem numVertices = G.numberOfNodes();

    std::vector<GraphElem> edgeCount(numVertices + 1);
    std::vector<GraphElemTuple> edgeList;
  
    std::cout << "Generating ER graph of numvertices: " << numVertices <<
      ", numEdges: " << numEdges << std::endl;

    GraphWeight weight = 1.0;

    G.forEdges([&](NetworKit::node u, NetworKit::node v) {
            assert((u >= 0) && (u < numVertices));
            assert((v >= 0) && (v < numVertices));

            if (randomEdgeWeight)
                weight = (GraphWeight)genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);

            edgeList.emplace_back(u, v, weight);
            edgeList.emplace_back(v, u, weight);

            edgeCount[u+1]++;
            edgeCount[v+1]++;
    });

    numEdges *= 2;
    g = new Graph(numVertices, numEdges);

    processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);
    t2 = mytimer(); 
    
    std::cout << "Total time to generate ER graph with N = " << N << 
        " and p = " << p << ": " << (t2-t1) << " secs\n";
}

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

#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include <utility>
#include <random>

#include "simple.hpp"

// Note: The difference between simple and simple2 is that
// `simple` considers the input graph as directed and only
// stores edges as they are listed in the file. Whereas
// `simple2` considers an undirected graph, and stores 
// both combinations of edge pairs.
void loadSimpleFileUn(Graph *&g, const std::string &fileName, bool indexOneBased, Weight_t wtype)
{
  std::ifstream ifs;

  double t0, t1, t2, t3;

  t0 = mytimer();

  ifs.open(fileName.c_str(), std::ifstream::in);
  if (!ifs) {
    std::cerr << "Error opening Simple format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string line;
  GraphElem maxVertex = -1, numEdges = 0, numVertices, skipLines = 0;

  do {
    GraphElem v0, v1;
    GraphWeight w = 0.0;

    std::getline(ifs, line);

    if(line[0]=='#' || line[0]=='%') {
        skipLines++;
        continue;
    }

    std::istringstream iss(line);
    if (wtype == ORG_WEIGHT || wtype == ABS_WEIGHT)
        iss >> v0 >> v1 >> w;
    else
        iss >> v0 >> v1;

    if (indexOneBased) {
        v0--; 
        v1--;
    }

    if (v0 > maxVertex)
      maxVertex = v0;
    if (v1 > maxVertex)
      maxVertex = v1;

    numEdges++;
  } while (!ifs.eof());

  numEdges--;  // Do not consider last line

  numVertices = maxVertex + 1;

  t1 = mytimer();

  std::cout << "Loading Simple file: " << fileName << ", numvertices: " << numVertices <<
    ", numEdges: " << numEdges << std::endl;
  std::cout << "Edge & vertex count time: " << (t1 - t0) << std::endl;

  t2 = mytimer();

  ifs.close();
  
  // file bounds known, start reading data
  ifs.open(fileName.c_str(), std::ifstream::in);

  std::vector<GraphElem> edgeCount(numVertices + 1);
  std::vector<GraphElemTuple> edgeList;

  for (GraphElem i = 0; i < numEdges+skipLines; i++) {
    GraphElem v0, v1;
    GraphWeight w = 0.0;

    std::getline(ifs, line);
    if(line[0]=='#' || line[0]=='%')		
        continue;
    
    std::istringstream iss(line);
 
    if (wtype == ORG_WEIGHT || wtype == ABS_WEIGHT)
        iss >> v0 >> v1 >> w;
    else
        iss >> v0 >> v1;

    if (indexOneBased) {
        v0--; 
        v1--;
    }
     
    if (wtype == ONE_WEIGHT)
        w = 1.0;

    if (wtype == RND_WEIGHT)
        w = genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);
    
    if (wtype == ABS_WEIGHT)
        w = std::fabs(w);

    edgeList.push_back({v0, v1, w});
    edgeList.push_back({v1, v0, w});

    edgeCount[v0+1]++;
    edgeCount[v1+1]++;
  }

  ifs.close();

  t3 = mytimer();
  
  numEdges = edgeList.size();

  std::cout << "Edge read time: " << (t3 - t2) << std::endl;

  t2 = mytimer();

  std::cout << "Before allocating graph" << std::endl;

  g = new Graph(numVertices, numEdges);

  assert(g);

  std::cout << "Allocated graph: " << g << std::endl;

  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);

  t3 = mytimer();  

  std::cout << "Graph populate time: " << (t3 - t2) << std::endl;
  std::cout << "Total I/O time: " << (t3 - t0) << std::endl;
} // loadSimpleFile

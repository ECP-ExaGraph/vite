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

#include "simple.hpp"

/// Assuming a file with just an edge list (directed)
void loadSimpleFile(Graph *&g, const std::string &fileName, 
        bool indexOneBased, Weight_t wtype)
{
  std::ifstream ifs;

  double t0, t1;
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

      std::getline(ifs, line);
      if(line[0] == '#' || line[0] == '%') {
          skipLines++;
          continue;
      }

      std::istringstream iss(line);
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

  numEdges--;
  numVertices = maxVertex + 1;
  
  std::cout << "Loading simple format file (directed edge-list): " 
      << fileName << ", numvertices: " << numVertices 
      << ", numEdges: " << numEdges << std::endl;

  ifs.close();
   
  // read the data
  ifs.open(fileName.c_str(), std::ifstream::in);

  std::vector<GraphElem> edgeCount(numVertices + 1);
  std::vector<GraphElemTuple> edgeList;
  
  for (GraphElem i = 0; i < numEdges+skipLines; i++) {
    
    GraphElem v0, v1;
    GraphWeight w;

    std::getline(ifs, line);
          
    if(line[0] == '#' || line[0] == '%') 
        continue;

    std::istringstream iss(line);
    iss >> v0 >> v1;
          
    if (indexOneBased) {
        v0--; 
        v1--;
    }
    
    if (wtype == ONE_WEIGHT)
        w = 1.0;

    if (wtype == RANDOM_WEIGHT)
        w = genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);

    edgeList.push_back({v0, v1, w});
    edgeCount[v0+1]++;
  }

  numEdges = edgeList.size();

  ifs.close();

  g = new Graph(numVertices, numEdges);
  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);

  t1 = mytimer();  

  std::cout << "Total graph processing time: " << (t1 - t0) << std::endl;
} // loadSimpleFile

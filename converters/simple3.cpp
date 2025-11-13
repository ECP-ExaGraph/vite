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
#include <map>

#include "simple.hpp"

/// Assuming a file with just an edge list (directed)
/// entries above threshold will be dropped
/// assumes zero-based indexing
void loadSimpleFileStr(Graph *&g, const std::string &fileName, GraphWeight threshold)
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
  GraphElem maxVertex = -1, numEdges = 0, numVertices = 0, skipLines = 0;
  std::map<std::string, GraphElem> tokens;
  
  // create map of tokens
  do {
      std::string u_key, v_key;
      GraphWeight w = 0.0;

      std::getline(ifs, line);
      
      if(line[0] == '#' || line[0] == '%') {
          skipLines++;
          continue;
      }

      std::istringstream ss(line);
      ss >> u_key >> v_key >> w;
 
      if (w <= threshold) {
          auto is_ok = tokens.insert({u_key, numVertices}); 
          if (is_ok.second)
             numVertices++;
          numEdges++;
      }
  } while (!ifs.eof());

  std::cout << "Loading Simple file: " << fileName << ", unique #vertices: " << numVertices << std::endl;

  ifs.close();
  
  // reopen file for reading data
  ifs.open(fileName.c_str(), std::ifstream::in);
  std::vector<GraphElem> edgeCount(numVertices);
  std::vector<GraphElemTuple> edgeList;

  do {
      GraphWeight w = 0.0;
      std::string u_key, v_key;

      std::getline(ifs, line);
      
      if(line[0] == '#' || line[0] == '%') {
          skipLines++;
          continue;
      }

      std::istringstream iss(line);
      iss >> u_key >> v_key >> w;

      if (w <= threshold) {
	      const GraphElem u = tokens[u_key];
	      const GraphElem v = tokens[v_key];

	      edgeList.push_back({u, v, w});
	      edgeCount[u+1]++;
      }
  } while (!ifs.eof());

  ifs.close();
   
  assert(numEdges == edgeList.size());

  ifs.close();
  std::cout << "Number of edges: " << numEdges << std::endl;

  g = new Graph(numVertices, numEdges);
  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);

  t1 = mytimer();  

  std::cout << "Total graph processing time: " << (t1 - t0) << std::endl;
} // loadSimpleFile

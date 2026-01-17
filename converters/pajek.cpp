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

#include "pajek.hpp"

void loadPajekFile(Graph *&g, const std::string &fileName, bool indexOneBased, Weight_t wtype)
{
  std::ifstream ifs;

  double t0, t1;

  t0 = mytimer();

  ifs.open(fileName.c_str(), std::ifstream::in);
  if (!ifs) {
    std::cerr << "Error opening Pajek format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  GraphElem numEdges = 0, numVertices = 0;
  
  /// http://mrvar.fdv.uni-lj.si/pajek/DrawEPS.htm
  // *Arcs are directed whereas *Edges indicate undirected
 
  std::string line;
  std::string target_word_vertices = "*Vertices";
  std::string target_word_arcs = "*Arcs";
  std::string target_word_edges = "*Edges";
  bool is_arcs = false, is_edges = false, is_vertices = false;
  
  do {
    if (line.find(target_word_vertices) != std::string::npos) {
       is_vertices = true;
       numVertices = std::stol(line.substr(line.find(' ') + 1));
    }

    if (line.find(target_word_arcs) != std::string::npos)
       is_arcs = true;
    if (line.find(target_word_edges) != std::string::npos)
       is_edges = true;

    if( is_arcs || is_edges)
        break;
  } while (std::getline(ifs, line));

  if (ifs.eof()) {
    std::cerr << "Error searching for arcs/edges in Pajek format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }
  
  if (!numVertices) {
    std::cerr << "Unable to find *Vertices in Pajek format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  std::vector<GraphElem> edgeCount;
  std::vector<GraphElemTuple> edgeList;

  edgeCount.resize(numVertices + 1, 0);

  while (std::getline(ifs, line))
  {
    GraphElem v0, v1;
    GraphWeight w = 1.0;

    std::istringstream iss(line);
    if (wtype == ORG_WEIGHT || wtype == ABS_WEIGHT)
        iss >> v0 >> v1 >> w;
    else
        iss >> v0 >> v1;

    if (indexOneBased) {
        v0--; 
        v1--;
    }
    
    if (wtype == RND_WEIGHT)
        w = genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);
    
    if (wtype == ABS_WEIGHT)
        w = std::fabs(w);

    edgeList.push_back({v0, v1, w});
    edgeCount[v0+1]++;

    if (is_arcs) {
       edgeList.push_back({v1, v0, w});
       edgeCount[v1+1]++;
    }
  }

  numEdges = edgeList.size();

  std::cout << "Loading Pajek file: " << fileName << ", #Vertices: " << numVertices << ", #Edges: " << numEdges << std::endl;

  ifs.close();
  
  g = new Graph(numVertices, numEdges);
  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);
  
  t1 = mytimer();  
  std::cout << "Total graph processing time: " << (t1 - t0) << std::endl;
} // loadPajekFile

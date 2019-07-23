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

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>

#include "dimacs.hpp"

/* Assuming `a u v w` format for both directed/undirected */

// reading directed files, edges stored twice
void loadDimacsFile(Graph *&g, const std::string &fileName, Weight_t wtype)
{
  std::ifstream ifs;

  ifs.open(fileName.c_str());
  if (!ifs) {
    std::cerr << "Error opening Dimacs format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string line;
  char comment;

  do {
    std::getline(ifs, line);
    comment = line[0];
  } while (comment == 'c');

  GraphElem numVertices, numEdges, value = 0;
  {
    std::string p, sp;

    std::istringstream iss(line);

    iss >> p >> sp >> numVertices >> numEdges;
    if (!iss || iss.fail() || p[0] != 'p') {
      std::cerr << "Error parsing DIMACS format: " << line << std::endl;
      std::cerr << "p: " << p << ", p[0]: " << p[0] << ", sp: " << sp <<
	", numVertices: " << numVertices << ", numEdges: " << numEdges << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  std::cout << "Loading (undirected) DIMACS file: " << fileName << ", initial numvertices: " << numVertices <<
    ", numEdges: " << numEdges << std::endl;

  double t0 = mytimer();

  std::vector<GraphElem> edgeCount(numVertices + 1);
  std::set<GraphElemTuple> edgeSets;
 
  // populate edge set
  for (GraphElem i = 0; i < numEdges; i++) {
    
    std::string label;
    GraphElem source, dest;
    GraphWeight weight;

    ifs >> label >> source >> dest >> weight;

    if (wtype == ONE_WEIGHT)
        weight = 1.0;

    if (wtype == RND_WEIGHT)
        weight = genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);
    
    if (wtype == ABS_WEIGHT)
        weight = std::fabs(weight);

    if (!ifs || ifs.fail()) {
      std::cerr << "Error parsing DIMACS edge" << std::endl;
      exit(EXIT_FAILURE);
    }

    edgeSets.insert({source-1, dest-1, weight});
    edgeSets.insert({dest-1, source-1, weight});
  }

  ifs.close();
  
  double t1 = mytimer();
  std::cout << "Time taken to read the file: " << (t1-t0) << " secs." << std::endl;

  for (auto it = edgeSets.begin(); it != edgeSets.end(); ++it)
      edgeCount[it->i_+1]++;

  std::vector<GraphElemTuple> edgeList(edgeSets.begin(), edgeSets.end());

  // adjust numEdges and edgeList size
  numEdges = edgeSets.size();
  std::cout << "Adjusted numEdges after removing duplicates: " << numEdges << std::endl; 
 
  // create graph data structure
  g = new Graph(numVertices, numEdges);
  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);
} // loadDimacsFile

// reading undirected files
void loadDimacsFileUn(Graph *&g, const std::string &fileName, Weight_t wtype)
{
  std::ifstream ifs;

  ifs.open(fileName.c_str());
  if (!ifs) {
    std::cerr << "Error opening Dimacs format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string line;
  char comment;

  do {
    std::getline(ifs, line);
    comment = line[0];
  } while (comment == 'c');

  GraphElem numVertices, numEdges, value = 0;
  {
    std::string p, sp;

    std::istringstream iss(line);

    iss >> p >> sp >> numVertices >> numEdges;
    if (!iss || iss.fail() || p[0] != 'p') {
      std::cerr << "Error parsing DIMACS format: " << line << std::endl;
      std::cerr << "p: " << p << ", p[0]: " << p[0] << ", sp: " << sp <<
	", numVertices: " << numVertices << ", numEdges: " << numEdges << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  std::cout << "Loading (undirected) DIMACS file: " << fileName << ", initial numvertices: " << numVertices <<
    ", numEdges: " << numEdges << std::endl;

  double t0 = mytimer();

  std::vector<GraphElem> edgeCount(numVertices + 1);
  std::set<GraphElemTuple> edgeSets;
 
  // populate edge  set
  for (GraphElem i = 0; i < numEdges; i++) {
    std::string label;
    GraphElem source, dest;
    GraphWeight weight;

    ifs >> label >> source >> dest >> weight;

    if (!ifs || ifs.fail()) {
      std::cerr << "Error parsing DIMACS edge" << std::endl;
      exit(EXIT_FAILURE);
    }
    
    if (wtype == ONE_WEIGHT)
        weight = 1.0;

    if (wtype == RND_WEIGHT)
        weight = genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);
    
    if (wtype == ABS_WEIGHT)
        weight = std::fabs(weight);

    edgeSets.insert({source-1, dest-1, weight});
  }

  ifs.close();
  
  double t1 = mytimer();
  std::cout << "Time taken to read the file: " << (t1-t0) << " secs." << std::endl;

  for (auto it = edgeSets.begin(); it != edgeSets.end(); ++it)
      edgeCount[it->i_+1]++;

  std::vector<GraphElemTuple> edgeList(edgeSets.begin(), edgeSets.end());

  // adjust numEdges and edgeList size
  numEdges = edgeSets.size();
  std::cout << "Adjusted numEdges after removing duplicates: " << numEdges << std::endl; 
 
  // create graph data structure
  g = new Graph(numVertices, numEdges);
  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);
} // loadDimacsFileUn

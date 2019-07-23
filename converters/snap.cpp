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
#include <limits>
#include <map>
#include <cmath>

#include "snap.hpp"

// Nodes have explicit (and arbitrary) node ids. There is no restriction for node 
// ids to be contiguous integers starting at 0. In TUNGraph and TNGraph edges have no 
// explicit ids -- edges are identified by a pair node ids.
void loadSNAPFile(Graph *&g, const std::string &fileName, Weight_t wtype)
{
  std::ifstream ifs;

  double t0, t1, t2, t3;

  t0 = mytimer();

  ifs.open(fileName.c_str(), std::ifstream::in);
  if (!ifs) {
    std::cerr << "Error opening SNAP format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string line;
  GraphElem numEdges = 0, numVertices;

  //Parse the comment lines for problem size
  size_t place = 0;
  while(1) { 
      std::getline(ifs, line);
      //Check if this line has problem sizes
      if (line[0] == '#')  { 
          std::size_t found_nodes = line.find("Nodes");
          if (found_nodes != std::string::npos) { // found our line
              std::istringstream iss(line);
              std::string key, val;
              int c = 0;
              while(std::getline(iss, key, ':') >> val) {
                  // only interested in parsing nodes/edges
                  if (c == 0) 
                      numVertices = std::stol(val);
                  else
                      numEdges = std::stol(val);
                  c++;
              }
          }
          place = ifs.tellg();
      }
      else {
          ifs.seekg(place); // move back
          break;
      }
  }

  t1 = mytimer();

  std::cout << "Loading SNAP file: " << fileName << ", numvertices: " << numVertices <<
    ", numEdges: " << numEdges << std::endl;
  std::cout << "Time taken to open file and parse number of vertices/edges: " 
      << (t1 - t0) << std::endl;
  
  // start parsing the data in file
  t2 = mytimer();

  std::vector<GraphElem> edgeCount(numVertices+1);   
  std::vector<GraphElemTuple> edgeList(numEdges*2);  // each edge stored twice

  // complete reading the file 
  // compute the node renumbering by pushing nodeid on 
  // a map and updating a temp edgeList
  std::map<GraphElem, GraphElem> clusterLocalMap; //Renumber vertices contiguously from zero
  std::map<GraphElem, GraphElem>::iterator storedAlready;
  GraphElem numUniqueVertices = 0;

  // update numEdges, as the input graph is
  // assumed to be undirected and weights are 1
  GraphElem nnz = 0;
  for (GraphElem i = 0L; i < numEdges; i++) {
      GraphWeight weight;
      GraphElem v0, v1;
      ifs >> v0 >> v1;

      if (!ifs || ifs.fail()) {
          std::cerr << "Error parsing SNAP edge" << std::endl;
          exit(EXIT_FAILURE);
      }

      // for node_0
      storedAlready = clusterLocalMap.find(v0);      //Check if it already exists
      if( storedAlready != clusterLocalMap.end() ) { //Already exists
          v0 = storedAlready->second; //Renumber the cluster id
      } else {
          clusterLocalMap[v0] = numUniqueVertices; //Does not exist, add to the map
          v0 = numUniqueVertices; //Renumber the vertex id
          numUniqueVertices++;    //Increment the number
      }

      // for node_1
      storedAlready = clusterLocalMap.find(v1);         //Check if it already exists
      if( storedAlready != clusterLocalMap.end() ) {	//Already exists
          v1 = storedAlready->second; //Renumber the cluster id
      } else {
          clusterLocalMap[v1] = numUniqueVertices; //Does not exist, add to the map
          v1 = numUniqueVertices; //Renumber the vertex id
          numUniqueVertices++; //Increment the number
      }
      
      if (wtype == RND_WEIGHT)
          weight = genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);

      if (wtype == ONE_WEIGHT)
          weight = 1.0;

      edgeList.emplace_back(v0, v1, weight);
      edgeList.emplace_back(v1, v0, weight);

      // edge counts
      edgeCount[v0 + 1]++;
      edgeCount[v1 + 1]++;
      nnz += 2;
  }
  
  ifs.close();
 
  assert((numEdges * 2) == nnz);

  t3 = mytimer();
  std::cout << "Time taken to allocate edgeList/edgeCount and read from file: " 
      << (t3 - t2) << std::endl;
   
  numEdges = nnz;
 
  t2 = mytimer();
  g = new Graph(numVertices, numEdges);
  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);

  t3 = mytimer();  

  std::cout << "Graph populate time: " << (t3 - t2) << std::endl;
  std::cout << "Total I/O time: " << (t3 - t0) << std::endl;
} // loadSNAPFile

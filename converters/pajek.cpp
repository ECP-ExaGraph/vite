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
#include <map>

#include "pajek.hpp"

/// This loader assumes the input file to have the following format:
/// First line: *Vertices <>
/// Second line onward: u v (edge list)
/// see loadPajekFileOrig for loading Pajek files in original format
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
  std::getline(ifs, line);
       
  if (line.find("*Vertices") != std::string::npos) 
     numVertices = std::stol(line.substr(line.find(' ') + 1));

  if (numVertices > 0) {
       std::cout << "Reading edges from Pajek format file: " << fileName << " with #Vertices listed as: " << numVertices << std::endl;
  }
  else {
    std::cerr << "Unable to find *Vertices in the first line of Pajek format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }
 
  std::vector<short> markVertices(numVertices, 0);
  std::vector<GraphElemTuple> edgeList;
  std::vector<GraphElem> edgeCount;

  // TODO FIXME accept weights as third parameter per line
  // currently assume w=1.0
  do {
    std::getline(ifs, line);
    //line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

    GraphElem v0, v1;
    GraphWeight w = 1.0;
    
    std::istringstream iss(line);
    iss >> v0 >> v1;

    if (v0 >= numVertices || v1 >= numVertices) {
       std::cerr << "Error: vertex indices are larger than the #Vertices: " << v0 << ", " << v1 << std::endl;
       exit(EXIT_FAILURE);
    }
     
    if (indexOneBased) {
        v0--; 
        v1--;
    }
    
    if (wtype == RND_WEIGHT)
        w = genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);
    
    if (wtype == ABS_WEIGHT)
        w = std::fabs(w);

    edgeList.push_back({v0, v1, w});
    edgeList.push_back({v1, v0, w});
    markVertices[v0] = 1;
    markVertices[v1] = 1;
  } while (!ifs.eof());
  
  ifs.close();

  std::map<GraphElem, GraphElem> vmap;
  GraphElem v_id = 0;
  
  for (GraphElem k = 0; k < numVertices; k++)
  {
    if (markVertices[k] == 1)
    {
      vmap.insert({k, v_id});
      v_id++;
    }
  }

  edgeCount.resize(v_id+1, 0);
  
  // adjust for duplicates 
  std::sort(edgeList.begin(), edgeList.end());
  auto last = std::unique(edgeList.begin(), edgeList.end());
  edgeList.erase(last, edgeList.end());
  
  /// adjust edge count/list to address gaps  
  for (auto el: edgeList) {
    el.i_  = vmap[el.i_];
    el.j_  = vmap[el.j_];
    edgeCount[el.i_+1]++;
  }
 
  numVertices = edgeCount.size()-1;
  numEdges    = edgeList.size(); 
  std::cout << "Legitimate {#Vertices, #Edges} recorded while reading the file: " << numVertices << ", " << numEdges << std::endl;

  g = new Graph(numVertices, numEdges);
  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);
  
  t1 = mytimer();  
  std::cout << "Total graph processing time: " << (t1 - t0) << std::endl;
} // loadPajekFile

void loadPajekFileOrig(Graph *&g, const std::string &fileName, bool indexOneBased, Weight_t wtype)
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
  bool found_edges = false;
      
  while(std::getline(ifs, line)) {
       if (line.find(target_word_vertices) != std::string::npos) 
          numVertices = std::stol(line.substr(line.find(' ') + 1));
       if (line.find("*Arcs") != std::string::npos || line.find("*Edges") != std::string::npos) {
          found_edges = true;
          break;
       } 
  }

  if (numVertices > 0 && found_edges) {
       std::cout << "Reading edges from Pajek format file: " << fileName << " with #Vertices listed as: " << numVertices << std::endl;
  }
  else {
    std::cerr << "Unable to find *Vertices and edge lists in Pajek format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }
 
  std::vector<short> markVertices(numVertices, 0);
  std::vector<GraphElemTuple> edgeList;
  std::vector<GraphElem> edgeCount;

  // TODO FIXME accept weights as third parameter per line
  // currently assume w=1.0
  do {
    
    std::getline(ifs, line);

    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

    GraphElem v0, v1;
    GraphWeight w = 1.0;
    std::istringstream iss(line);

    iss >> v0 >> v1;

    if (v0 >= std::numeric_limits<GraphElem>::max() || v1 >= std::numeric_limits<GraphElem>::max())
        continue;
    if (v0 >= numVertices || v1 >= numVertices) {
       std::cerr << "Error: vertex indices are larger than the #Vertices: " << v0 << ", " << v1 << std::endl;
       exit(EXIT_FAILURE);
    }
     
    if (indexOneBased) {
        v0--; 
        v1--;
    }
    
    if (wtype == RND_WEIGHT)
        w = genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);
    
    if (wtype == ABS_WEIGHT)
        w = std::fabs(w);

    edgeList.push_back({v0, v1, w});
    edgeList.push_back({v1, v0, w});
    markVertices[v0] = 1;
    markVertices[v1] = 1;
  } while (!ifs.eof());
  
  ifs.close();

  std::map<GraphElem, GraphElem> vmap;
  GraphElem v_id = 0;
  
  for (GraphElem k = 0; k < numVertices; k++)
  {
    if (markVertices[k] == 1)
    {
      vmap.insert({k, v_id});
      v_id++;
    }
  }

  edgeCount.resize(v_id+1, 0);
  
  // adjust for duplicates 
  std::sort(edgeList.begin(), edgeList.end());
  auto last = std::unique(edgeList.begin(), edgeList.end());
  edgeList.erase(last, edgeList.end());
  
  /// adjust edge count/list to address gaps  
  for (auto el: edgeList) {
    el.i_  = vmap[el.i_];
    el.j_  = vmap[el.j_];
    edgeCount[el.i_+1]++;
  }
 
  numVertices = edgeCount.size()-1;
  numEdges    = edgeList.size(); 
  std::cout << "Legitimate {#Vertices, #Edges} recorded while reading the file: " << numVertices << ", " << numEdges << std::endl;

  g = new Graph(numVertices, numEdges);
  processGraphData(*g, edgeCount, edgeList, numVertices, numEdges);
  
  t1 = mytimer();  
  std::cout << "Total graph processing time: " << (t1 - t0) << std::endl;
} // loadPajekFile

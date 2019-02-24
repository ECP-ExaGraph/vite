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
#include <sstream>

#include "metis.hpp"

void loadMetisFile(Graph *&g, const std::string &fileName, Weight_t wtype)
{
  std::ifstream ifs;

  ifs.open(fileName.c_str());
  if (!ifs) {
    std::cerr << "Error opening Metis format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string line;
  char comment;

  do {
    std::getline(ifs, line);
    comment = line[0];
  } while (comment == '%');

  GraphElem numVertices, numEdges, value = 0L;
  {
    std::istringstream iss(line);

    iss >> numVertices >> numEdges;
    if (!iss.eof())
      iss >> value;
  }
  
  numEdges *= 2; // Metis' edges are not directional

  std::cout << "Loading Metis file: " << fileName << ", numvertices: " << numVertices <<
    ", numEdges: " << numEdges << std::endl;

  g = new Graph(numVertices, numEdges);

  GraphElem edgePos = 0;

  g->setEdgeStartForVertex(0, 0);

  switch (value) {
  case 10:
    std::cout << "Metis format vertex weights ignored" << std::endl;
  case 0: // No weights in Metis file
    for (GraphElem i = 0; i < numVertices; i++) {
      GraphElem j = 0L, neighbor, oldNeighbor=-1;

      std::getline(ifs, line);
      std::istringstream iss(line);

      while (!iss.eof()) {
        iss >> neighbor;
      	if (!iss || iss.eof())
//          if((oldNeighbor == neighbor))
	        break;
        GraphWeight weight;
        oldNeighbor = neighbor;
      	j++;
      	Edge &edge = g->getEdge(edgePos);
        edge.tail = neighbor - 1;

        if (wtype == ONE_WEIGHT)
            weight = 1.0;

        if (wtype == RANDOM_WEIGHT)
            weight = (GraphWeight)genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);

        edge.weight = weight;
        //std::cout<<i <<","<<edge.tail<<std::endl;
	edgePos++;
      }
      g->setEdgeStartForVertex(i + 1L, edgePos);
    }
    break;
  case 11:
    std::cout << "Metis format vertex weights ignored" << std::endl;
  case 1:
    for (GraphElem i = 0; i < numVertices; i++) {
      GraphElem j = 0, neighbor;
      GraphWeight weight;

      std::getline(ifs, line);
      std::istringstream iss(line);

      while (!iss.eof()) {
	iss >> neighbor >> weight;
	if (!iss || iss.eof())
	  break;
        
        if (wtype == ONE_WEIGHT)
            weight = 1.0;

        if (wtype == RANDOM_WEIGHT)
            weight = (GraphWeight)genRandom(RANDOM_MIN_WEIGHT, RANDOM_MAX_WEIGHT);

	j++;
	Edge &edge = g->getEdge(edgePos);
	edge.tail = neighbor - 1;
	edge.weight = weight;
	edgePos++;
      }

      g->setEdgeStartForVertex(i + 1, edgePos);
    }
    break;
  default:
    std::cerr << "Inconsistent value for weight flag in Metis format: " << value << std::endl;
    exit(EXIT_FAILURE);
  }

  ifs.close();
} // loadMetisFile

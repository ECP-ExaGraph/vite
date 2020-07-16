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

#ifndef __GRAPH_H
#define __GRAPH_H

#include <cassert>

#include <algorithm>
#include <ostream>
#include <vector>

#include "edge.hpp"

struct Comm {
  GraphElem size;
  GraphWeight degree;

  Comm() : size(0), degree(0) { };
};


#if defined(__CRAY_MIC_KNL) && defined(USE_AUTOHBW_MEMALLOC)
#include <hbw_allocator.h>
typedef std::vector<Comm, hbw::allocator<Comm> > CommVector;
#else
typedef std::vector<Comm> CommVector;
#endif

class Graph {
protected:
  typedef std::vector<Edge> EdgeList;

  GraphElem numVertices;
  GraphElem numEdges;

public:
  EdgeIndexes edgeListIndexes;
  EdgeList edgeList;

  Graph(const GraphElem onv, const GraphElem one);
  Graph(const Graph &othis);
  ~Graph();

  GraphElem getNumVertices() const;
  GraphElem getNumEdges() const;
  void setEdgeWeightstoOne();
  void setNumEdges(GraphElem numEdges);
  void getEdgeRangeForVertex(const GraphElem vertex, GraphElem &e0, GraphElem &e1) const;
  const Edge &getEdge(const GraphElem edge) const;

  void setEdgeStartForVertex(const GraphElem vertex, const GraphElem e0);
  Edge &getEdge(const GraphElem edge);
  
  friend std::ostream &operator <<(std::ostream &os, const Graph &g);
protected:
  Graph();
  Graph &operator = (const Graph &othis);
};

inline Graph::Graph()
  : numVertices(0), numEdges(0)
{
} // Graph

inline Graph::Graph(const GraphElem onv, const GraphElem one)
  : numVertices(onv), numEdges(one)
{
  edgeListIndexes.resize(numVertices + 1);
  edgeList.resize(numEdges);

  std::for_each(edgeListIndexes.begin(), edgeListIndexes.end(),
		[] (GraphElem &idx) { idx = 0; } );
  
} // Graph

inline Graph::Graph(const Graph &othis)
  : numVertices(othis.numVertices), numEdges(othis.numEdges)
{
  edgeListIndexes.resize(numVertices + 1);
  edgeList.resize(numEdges);

  std::copy(othis.edgeListIndexes.begin(), othis.edgeListIndexes.end(),
	    edgeListIndexes.begin());
  std::copy(othis.edgeList.begin(), othis.edgeList.end(), edgeList.begin());
} // Graph

inline Graph::~Graph()
{
} // ~Graph

inline GraphElem Graph::getNumVertices() const
{
  return numVertices;
} // getNumVertices

inline GraphElem Graph::getNumEdges() const
{
  return numEdges;
} // getNumEdges

inline void Graph::setNumEdges(GraphElem numEdges) {
    this->edgeList.resize(numEdges);	
    this->numEdges=numEdges;
}

inline void Graph::setEdgeWeightstoOne() {
    for (GraphElem i = 0; i < this->numEdges; i++)
        this->edgeList[i].weight = 1.0;
}

inline void Graph::getEdgeRangeForVertex(const GraphElem vertex, GraphElem &e0, GraphElem &e1) const
{
  assert((vertex >= 0) && (vertex < numVertices));
#if defined(DEBUG_BUILD)
  e0 = edgeListIndexes.at(vertex);
  e1 = edgeListIndexes.at(vertex + 1);
#else
  e0 = edgeListIndexes[vertex];
  e1 = edgeListIndexes[vertex + 1];
#endif
} // getEdgeRangeForVertex

inline const Edge &Graph::getEdge(const GraphElem edge) const
{
#if defined(DEBUG_BUILD)
  assert((edge >= 0) && (edge < numEdges));
  return edgeList.at(edge);
#else
  return edgeList[edge];
#endif
} // getEdge

inline void Graph::setEdgeStartForVertex(const GraphElem vertex, const GraphElem e0)
{
#if defined(DEBUG_BUILD)
  assert((vertex >= 0) && (vertex <= numVertices));
  assert((e0 >= 0) && (e0 <= numEdges));
  edgeListIndexes.at(vertex) = e0;
#else
  edgeListIndexes[vertex] = e0;
#endif
} // setEdgeRangeForVertex

inline Edge &Graph::getEdge(const GraphElem edge)
{
#if defined(DEBUG_BUILD)
  if ((edge < 0) || (edge >= numEdges))
    std::cerr << "ERROR: out of bounds access: " << edge << ", max: " << numEdges << std::endl;
  assert((edge >= 0) && (edge < numEdges));
  return edgeList.at(edge);
#else
  return edgeList[edge];  
#endif
} // getEdge

inline std::ostream &operator <<(std::ostream &os, const Graph &g)
{
#if defined(DEBUG_BUILD)
  os << "Number of vertices: " << g.numVertices << ", number of edges: " << g.numEdges <<
    std::endl;
#endif

  for (GraphElem i = 0; i < g.numVertices; i++) {
#if defined(DEBUG_BUILD)
    const GraphElem lb = g.edgeListIndexes.at(i), ub = g.edgeListIndexes.at(i + 1);
    os << "Vertex: " << i << ", number of neighbors: " << ub - lb << std::endl;
#else
    const GraphElem lb = g.edgeListIndexes[i], ub = g.edgeListIndexes[i + 1];
#endif

    for (GraphElem j = lb; j < ub; j++) {
      const Edge &edge = g.getEdge(j);

#if defined(DEBUG_BUILD)
      os << "Edge to: " << edge.tail << ", weight: " << edge.weight << std::endl;
#endif
    }
  }

  return os;
} // operator <<

#endif // __GRAPH_H

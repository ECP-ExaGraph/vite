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

#ifndef __DISTGRAPH_H
#define __DISTGRAPH_H
///
#include <cassert>
#include <cmath>

#include <algorithm>

#include "graph.hpp"

// TODO FIXME purge this entire class 
// and just have one graph class like 
// miniVite

#define PI                          (3.14159)
#define SR_UP_TAG                   100
#define SR_DOWN_TAG                 101
#define SR_SIZES_UP_TAG             102
#define SR_SIZES_DOWN_TAG           103
#define SR_X_UP_TAG                 104
#define SR_X_DOWN_TAG               105
#define SR_Y_UP_TAG                 106
#define SR_Y_DOWN_TAG               107

typedef std::vector<GraphElem> PartRanges;

class DistGraph {
protected:
  GraphElem totalNumVertices;
  GraphElem totalNumEdges;
  Graph *localGraph;

public:
  DistGraph(const GraphElem tnv, const GraphElem tne);
  DistGraph(const DistGraph &othis);
  ~DistGraph();

  GraphElem getTotalNumVertices() const;
  GraphElem getTotalNumEdges() const;

  void createLocalGraph(const GraphElem lnv, const GraphElem lne,
			const PartRanges *oparts = NULL);
  Graph &getLocalGraph();
  const Graph &getLocalGraph() const;
  GraphElem getBase(const int me) const;
  GraphElem getBound(const int me) const;
  GraphElem localToGlobal(GraphElem idx, const int me) const;
  GraphElem globalToLocal(GraphElem idx, const int me) const;
  int getOwner(const GraphElem v) const;
  PartRanges *parts;
  void setNumEdges(GraphElem numEdges); 
  void printStats();
  GraphElem getNumGhosts(const int me) const;
protected:
  DistGraph();
  DistGraph &operator = (const DistGraph &othis);
};

void balanceEdges(int nprocs, std::string& fileName, std::vector<GraphElem>& mbins);
void loadDistGraphMPIIO(int me, int nprocs, int ranks_per_node, 
        DistGraph *&dg, std::string& fileName);
void loadDistGraphMPIIOBalanced(int me, int nprocs, int ranks_per_node, 
        DistGraph *&dg, std::string& fileName);

// graph generation
void generateInMemGraph(int rank, int nprocs, DistGraph *&dg, GraphElem nv, GraphWeight randomEdgePercent, std::string fileOut);
DistGraph* generateRGG(int rank, int nprocs, GraphElem nv, GraphWeight rn, GraphWeight randomEdgePercent, std::string fileOut);

void writeGraph(int me, int nprocs, DistGraph *&dg, std::vector<GraphElem>& edgeCount, std::string &fileName);

inline DistGraph::DistGraph()
  : totalNumVertices(0), totalNumEdges(0), localGraph(NULL), parts(NULL)
{
} // DistGraph

inline DistGraph::DistGraph(const GraphElem tnv, const GraphElem tne)
  : totalNumVertices(tnv), totalNumEdges(tne), localGraph(NULL), parts(NULL)
{
} // DistGraph

inline DistGraph::DistGraph(const DistGraph &othis)
  : totalNumVertices(othis.totalNumVertices), totalNumEdges(othis.totalNumEdges),
    localGraph(new Graph(*othis.localGraph)), parts(NULL)
{ parts = new PartRanges(*othis.parts); } // DistGraph

inline DistGraph::~DistGraph()
{
  if (localGraph)
    delete localGraph;
  delete parts;
} // ~DistGraph

inline GraphElem DistGraph::getTotalNumVertices() const
{ return totalNumVertices; } // getTotalNumVertices

inline GraphElem DistGraph::getTotalNumEdges() const
{ return totalNumEdges; } // getTotalNumEdges

// print statistics about edge distribution
inline void DistGraph::printStats()
{
    int me, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    Graph &g = this->getLocalGraph(); // local graph 
    long lne = (long)g.getNumEdges(); // local #edges
    long nv = (long)this->getTotalNumVertices(); // global #vertices
    // TODO FIXME currently totalNumEdges variable stores local
    // number of edges and not total, keep a separate variable
    //const GraphElem ne = this->getTotalNumEdges(); // global #edges
    long ne = 0;
    MPI_Allreduce(&lne, &ne, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    long sumdeg = 0, maxdeg = 0, mindeg = 0;
    MPI_Reduce(&lne, &sumdeg, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lne, &maxdeg, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lne, &mindeg, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);

    long my_sq = lne*lne;
    long sum_sq = 0;
    MPI_Reduce(&my_sq, &sum_sq, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double average  = (double) sumdeg / size;
    double avg_sq   = (double) sum_sq / size;
    double var      = avg_sq - (average*average);
    double stddev   = sqrt(var);

    MPI_Barrier(MPI_COMM_WORLD);

    if (me == 0)
    {
      std::cout << std::endl;
      std::cout << "-------------------------------------------------------" << std::endl;
      std::cout << "Graph edge distribution characteristics" << std::endl;
      std::cout << "-------------------------------------------------------" << std::endl;
      std::cout << "Number of vertices: " << nv << std::endl;
      std::cout << "Number of edges: " << ne << std::endl;
      std::cout << "Maximum number of edges: " << maxdeg << std::endl;
      std::cout << "Average number of edges: " << average << std::endl;
      std::cout << "Minimum number of edges: " << mindeg << std::endl;
      std::cout << "Expected value of X^2: " << avg_sq << std::endl;
      std::cout << "Variance: " << var << std::endl;
      std::cout << "Standard deviation: " << stddev << std::endl;
      std::cout << "-------------------------------------------------------" << std::endl;

    }
}

inline void DistGraph::createLocalGraph(const GraphElem lnv, const GraphElem lne,
					const PartRanges *oparts)
{
#ifdef DEBUG_PRINTF    
  assert(!localGraph);
#endif

  localGraph = new Graph(lnv, lne);
  parts = new PartRanges(*oparts);
} // createLocalGraph

inline Graph &DistGraph::getLocalGraph()
{
#ifdef DEBUG_PRINTF    
  assert(localGraph);
#endif

  return *localGraph;
} // getLocalGraph

inline const Graph &DistGraph::getLocalGraph() const
{
#ifdef DEBUG_PRINTF    
  assert(localGraph);
#endif

  return *localGraph;
} // getLocalGraph

inline GraphElem DistGraph::getBase(const int me) const
{ 
#ifdef DEBUG_PRINTF    
    return parts->at(me);
#else
    return parts->operator[](me);
#endif
} // getBase

inline GraphElem DistGraph::getBound(const int me) const
{
#ifdef DEBUG_PRINTF    
    return parts->at(me + 1);
#else
    return parts->operator[](me + 1);
#endif
} // getBound
  
inline GraphElem DistGraph::localToGlobal(GraphElem idx, const int me) const
{ 
    return (idx + getBase(me)); 
} // localToGlobal
        
inline GraphElem DistGraph::globalToLocal(GraphElem idx, const int me) const
{ 
    return (idx - getBase(me)); 
} // globalToLocal
 
inline void DistGraph::setNumEdges(const GraphElem numEdges)
{ this->totalNumEdges=numEdges; }

inline int DistGraph::getOwner(const GraphElem v) const
{
#ifdef DEBUG_PRINTF    
    assert((v >= 0) && (v < totalNumVertices));
#endif
    const PartRanges::const_iterator iter = std::upper_bound(parts->begin(), parts->end(), v);

#ifdef DEBUG_PRINTF    
    assert(iter != parts->end());
#endif
    return (iter - parts->begin() -1);
} // getOwner

inline GraphElem DistGraph::getNumGhosts(const int me) const
{
  const Graph &g = this->getLocalGraph();  
  GraphElem numGhosts = 0;  
  for (GraphElem i = 0; i < g.getNumVertices(); i++) {
    const GraphElem lb = g.edgeListIndexes[i], ub = g.edgeListIndexes[i + 1];
    for (GraphElem j = lb; j < ub; j++) {
      const Edge &edge = g.getEdge(j);
      if (this->getOwner(edge.tail) != me)
          numGhosts += 1;
    }
  }
  return numGhosts;
} // return number of ghost vertices
#endif // __DISTGRAPH_H

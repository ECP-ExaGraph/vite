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

#ifndef __BUILD_NEXT_PHASE_H
#define __BUILD_NEXT_PHASE_H

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <iostream>
#include <numeric>

#include <omp.h>

#include "edge.hpp"
#include "louvain.hpp"

typedef struct edgeInfo{
  GraphElem s;
  GraphElem t;
  GraphWeight w;
}EdgeInfo;

static MPI_Datatype edgeType;

#if defined(__CRAY_MIC_KNL) && defined(USE_AUTOHBW_MEMALLOC)
typedef std::vector<EdgeInfo, hbw::allocator<EdgeInfo> > EdgeVector;
typedef std::unordered_set<GraphElem, std::hash<GraphElem>, std::equal_to<GraphElem>, 
	hbw::allocator<GraphElem> > RemoteCommList;
typedef std::vector<RemoteCommList, hbw::allocator<RemoteCommList>> PartArray;
typedef std::map<GraphElem, GraphWeight, std::less<GraphElem>, 
	hbw::allocator< std::pair< const GraphElem, GraphWeight > > > NewEdge;
typedef std::unordered_map<GraphElem, NewEdge, std::hash<GraphElem>, std::equal_to<GraphElem>, 
	hbw::allocator< std::pair< const GraphElem, NewEdge > > > NewEdgesMap;
#else
typedef std::vector<EdgeInfo> EdgeVector;
typedef std::unordered_set<GraphElem> RemoteCommList;
typedef std::vector<RemoteCommList> PartArray;
typedef std::map<GraphElem, GraphWeight> NewEdge;
typedef std::unordered_map<GraphElem,NewEdge> NewEdgesMap;
#endif

void createEdgeMPIType();
void destroyEdgeMPIType();

static GraphElem distReNumber(int nprocs, ClusterLocalMap& lookUp, int me, 
        DistGraph &dg, const size_t &ssz, const size_t &rsz, 
        const std::vector<GraphElem> &ssizes, const std::vector<GraphElem> &rsizes, 
        const std::vector<GraphElem> &svdata, const std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, VertexCommMap &remoteComm);

void fill_newEdgesMap(int me, NewEdgesMap &newEdgesMap, 
        DistGraph& dg, CommunityVector &cvect, VertexCommMap &remoteComm, 
        ClusterLocalMap &lookUp);

void send_newEdges(int me, int nprocs, DistGraph* &dg, GraphElem newGlobalNumVertices,
  NewEdgesMap& newEdgesMap);

void distbuildNextLevelGraph(int nprocs, int me, DistGraph*& dg, 
        const size_t &ssz, const size_t &rsz, const std::vector<GraphElem> &ssizes, 
        const std::vector<GraphElem> &rsizes, const std::vector<GraphElem> &svdata, 
        const std::vector<GraphElem> &rvdata, CommunityVector &cvect);
#endif

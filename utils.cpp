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
//               Ananth Kalyanaraman (ananth@eecs.wsu.edu)
//               Hao Lu (luhowardmark@wsu.edu)
//               Sayan Ghosh (sayan.ghosh@wsu.edu)
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

#include <sys/resource.h>
#include <sys/time.h>

#include <cstdlib>
#include <iostream>
#include <numeric>

#include "utils.hpp"

double mytimer(void)
{
  static long int start = 0L, startu;

  const double million = 1000000.0;

  timeval tp;

  if (start == 0L) {
    gettimeofday(&tp, NULL);

    start = tp.tv_sec;
    startu = tp.tv_usec;
  }

  gettimeofday(&tp, NULL);

  return (static_cast<double>(tp.tv_sec - start) + (static_cast<double>(tp.tv_usec - startu) /
						    million));
}

bool isDigits(const std::string& str) 
{
    for (unsigned char c : str) { 
        if (!std::isdigit(c)) { 
            return false; 
        }
    }
    return true; 
}

// Random number generator from B. Stroustrup: 
// http://www.stroustrup.com/C++11FAQ.html#std-random
GraphWeight genRandom(GraphWeight low, GraphWeight high)
{
    static std::default_random_engine re {};
    using Dist = std::uniform_real_distribution<GraphWeight>;
    static Dist uid {};
    return uid(re, Dist::param_type{low,high});
}

void processGraphData(Graph &g, std::vector<GraphElem> &edgeCount,
		      std::vector<GraphElemTuple> &edgeList,
		      const GraphElem nv, const GraphElem ne)
{
  std::vector<GraphElem> ecTmp(nv + 1);

  std::partial_sum(edgeCount.begin(), edgeCount.end(), ecTmp.begin());
  edgeCount = ecTmp;

  g.setEdgeStartForVertex(0, 0);

  for (GraphElem i = 0; i < nv; i++)
    g.setEdgeStartForVertex(i + 1, edgeCount[i + 1]);

  edgeCount.clear();

  auto ecmp = [] (GraphElemTuple const& e0, GraphElemTuple const& e1)
  { return ((e0.i_ < e1.i_) || ((e0.i_ == e1.i_) && (e0.j_ < e1.j_))); };
  
  if (!std::is_sorted(edgeList.begin(), edgeList.end(), ecmp)) {
#if defined(DEBUG_PRINTF)
    std::cout << "Edge list is not sorted" << std::endl;
#endif
    std::sort(edgeList.begin(), edgeList.end(), ecmp);
  }
  else {
#if defined(DEBUG_PRINTF)
    std::cout << "Edge list is sorted!" << std::endl;
#endif
  }

  GraphElem ePos = 0;
  for (GraphElem i = 0; i < nv; i++) {
	  GraphElem e0, e1;
	  g.getEdgeRangeForVertex(i, e0, e1);

	  if ((i % 100000) == 0)
		  std::cout << "Processing edges for vertex: " << i << ", range(" << e0 << ", " << e1 <<
			  ")" << std::endl;

	  for (GraphElem j = e0; j < e1; j++) {
		  Edge &edge = g.getEdge(j);

		  assert(ePos == j);
		  assert(i == edgeList[ePos].i_);
		  edge.tail = edgeList[ePos].j_;
		  edge.weight = edgeList[ePos].w_;

		  ePos++;
	  }
  }
} // processGraphData

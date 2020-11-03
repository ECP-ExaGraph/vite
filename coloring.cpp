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
#include "coloring.hpp"

ColorElem distColoringMultiHashMinMax(const int me, const int nprocs, const DistGraph &dg, ColorVector &vertexColor, const ColorElem nHash, const int target_percent, const bool singleIteration)
{

 std::vector<GraphElem> ghostVertices;
 std::vector<GraphElem> ghostSizes(nprocs);

 ColoredVertexSet remoteColoredVertices; 

 unsigned int seed = 1012;
 ColorElem nextColor=0; 

 GraphElem lastCount=0;
 GraphElem currentCount=0;

 const Graph &g = dg.getLocalGraph();
 const GraphElem lnv = g.getNumVertices();
 const GraphElem tnv = dg.getTotalNumVertices(); 

 GraphElem totalUnassigned = tnv;

 GraphElem targetCount = ( tnv * target_percent ) / 100;

 bool finished = false;

 // resize and initialize
 vertexColor.resize(lnv);

 for (GraphElem i=0; i<lnv; i++)
	vertexColor[i]=-1;

 // if it is not a single iteration, set up the list of ghost vertices
 if (!singleIteration) 
	setUpGhostVertices( me, nprocs, dg, ghostVertices, ghostSizes);


  // cycle until target not meet
  while (!finished) {
		
        distColoringIteration(me, dg, vertexColor, remoteColoredVertices, nHash, nextColor, seed);

        if (!singleIteration) {
	        sendColoredRemoteVertices(me, nprocs, dg, remoteColoredVertices, vertexColor, ghostVertices, ghostSizes); 
	}  

	 totalUnassigned = countUnassigned(vertexColor);
	 currentCount = tnv - totalUnassigned;       

	finished = singleIteration == true || currentCount >= targetCount || lastCount==currentCount; 

	nextColor += nHash * 2;
	seed = hash(seed, 0);
	lastCount=currentCount;	
  }

  if (singleIteration)
      setUpGhostVertices(me,nprocs,dg,ghostVertices, ghostSizes);

#if defined(CHECK_COLORING_CONFLICTS)
        GraphElem conflicts;
        conflicts = distCheckColoring(me, nprocs, dg, vertexColor, ghostVertices, ghostSizes);
        if (conflicts>0) 
            std::cout << " ERROR!!! Bad coloring !!! " << std::endl;	
#endif

return nextColor; 

}

unsigned int hash(unsigned int a, unsigned int seed)
{
  a ^= seed;
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) + (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a ^ 0xd3a2646c) + (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) + (a >> 16);

  return a;
}

void distColoringIteration(const int me, const DistGraph &dg, ColorVector &vertexColor, ColoredVertexSet &remoteColoredVertices, const ColorElem nHash, const ColorElem nextColor, const unsigned int seed)
{
	const Graph &g = dg.getLocalGraph();
	const GraphElem base = dg.getBase(me); 
	const GraphElem bound = dg.getBound(me);
        
         const GraphElem nv = g.getNumVertices();
         const GraphElem ne = g.getNumEdges();
  	
	
       	for (GraphElem v=0; v < nv; v++) {
	
  	        ColorElem possible_colors = 2 * nHash;
	        unsigned int vHash[nHash];

		if(vertexColor[v] != -1)
				continue;
	
		int not_min = 0, not_max=0;

		bool can_color =true;

		for (ColorElem t=0; t<nHash; t++)
		{
			vHash[t]= hash(v+base, seed+1043*t);
		}	
		
		GraphElem e0, e1;
		
		g.getEdgeRangeForVertex(v, e0, e1);

		for(long k = e0; k < e1; k++ ){
		
 			const Edge &edge = g.getEdge(k);
			
			if(v+base == edge.tail ) //Ignore Self-loops
				continue;
						
			// neighbor is local
			if (edge.tail>=base && edge.tail < bound) {
				if ((vertexColor[edge.tail-base]) != -1 && (vertexColor[edge.tail-base] < nextColor)) {
#ifdef DEBUG_COLORING
					std::cout << "Did you skip ??? BaseColor this iteration is: " << nextColor << std::endl; 
#endif					
                                        continue;

				}
		 	}
			else { // is remote, check if it has been colored
			  ColoredVertexSet::const_iterator it = remoteColoredVertices.find(edge.tail);
			  // if it is in the colored set, we ignore.
			  if (it!=remoteColoredVertices.end()) {
#ifdef DEBUG_COLORING
                              std::cout << "Did you skip ??? BaseColor this iteration is: " << nextColor << std::endl; 
                              std::cout << " Colored " << edge.tail << " BaseColor this iteration is: " << nextColor << std::endl;
#endif					
				continue;
				}
			}	

			// for each hash
			for (ColorElem t=0; t<nHash; t++)
			{
				unsigned int jHash = hash(edge.tail, seed+1043*t);
				
				if (vHash[t] <= jHash && !(not_max & (0x1 << t))) {
					not_max |= (0x1 << t);
					possible_colors--;
				}
				
				if (vHash[t] >= jHash && !(not_min & (0x1 << t)))
				{
					not_min |= (0x1 << t);
					possible_colors--;
				}
			}	


			if (possible_colors == 0){
				can_color=false;
				break;				
 			}
		}
	
		if (can_color) {
			// pick color

			ColorElem col_id = (v+base) % possible_colors;
			ColorElem this_col_id = 0;

			for (ColorElem t=0; t <nHash; t++)
			{
				if (!(not_min & (0x1 << t)) && col_id == this_col_id) {
					vertexColor[v]= 2 * t + nextColor;
		//			std::cout << "not min - me " << me << " vertex: " << v+base << " VertexColor: " << vertexColor[v] << std::endl;
					break;
				}
			
				this_col_id += !(not_min & (0x1 << t));
				
				if (!(not_max & (0x1 << t)) && col_id == this_col_id) {
					vertexColor[v]=2*t+1+nextColor;
		//			std::cout << "not max - me " << me << " vertex: " << v+base << " VertexColor: " << vertexColor[v] << std::endl;
	
					break;
				}	

				this_col_id += !(not_max & (0x1 << t)); 
//				std::cout << "Proc " << me << " Vertex " << v << "this col id " << this_col_id << std::endl;
			}//End of for(ihash)
		}	

	}

}

// Exchange ghost vertices requests
void setUpGhostVertices(const int me, const int nprocs, const DistGraph &dg, std::vector<GraphElem> &ghostVertices, std::vector<GraphElem> &ghostSizes)
{
	std::vector<std::unordered_set<GraphElem>> sendRemoteVertices(nprocs);
	std::vector<GraphElem> sRemoteVertices;

	const Graph &g = dg.getLocalGraph();
	const GraphElem tnv = dg.getTotalNumVertices();
	const GraphElem base = dg.getBase(me); 
	const GraphElem bound = dg.getBound(me);
        const GraphElem nv = g.getNumVertices();
        const GraphElem ne = g.getNumEdges();

 	GraphElem rsz = 0, ssz = 0;


        for (GraphElem i=0; i<nv; i++)
	{
		GraphElem e0, e1;

		g.getEdgeRangeForVertex(i, e0, e1);
	
		//For every remote neighbors
		//Find owner of remote neighbors, record a messages of that id
		
		for(GraphElem j = e0; j < e1; j++)
		{
			const Edge &edge = g.getEdge(j);
		
			const int owner = dg.getOwner(edge.tail);
			
				if (owner != me) {
				sendRemoteVertices[owner].insert(edge.tail);

					
			}
		}
	}

	//MPI Sending size of updates
	std::vector<GraphElem> ssizes(nprocs);
  	std::vector<MPI_Request> sreqs(nprocs), rreqs(nprocs);

  	for (int i = 0; i < nprocs; i++) {
		     	ssizes[i] = sendRemoteVertices[i].size();
	//		std::cout << " Me " << me << " Sending to " << i << " Size " << ssizes[i] << std::endl; 
		if (i != me){
      			MPI_Irecv(&ghostSizes[i], 1, MPI_GRAPH_TYPE, i, ColoringSizeTag, MPI_COMM_WORLD, &rreqs[i]);
	  		MPI_Isend(&ssizes[i], 1, MPI_GRAPH_TYPE, i, ColoringSizeTag, MPI_COMM_WORLD, &sreqs[i]);
    		}	
		else 	{
      			rreqs[i] = MPI_REQUEST_NULL;
      			sreqs[i] = MPI_REQUEST_NULL;
  		}
	}	

  	MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  	MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);

	int rpos=0;
        int spos=0;

	for (int i=0; i<nprocs; i++) {
	//	std::cout << " Me " << me << " receiving from " << i << " Size " << ghostSizes[i] << std::endl;

		rsz+=ghostSizes[i];
		ssz+=ssizes[i];
	}




        ghostVertices.resize(rsz);
	sRemoteVertices.resize(ssz);

  	for (int i = 0; i < nprocs; i++) {
		if (i != me){
#ifdef DEBUG_COLORING
			std::cout << " me: " << me << " from " << i << " size: " << ghostSizes[i] << std::endl;		
#endif      			
                        MPI_Irecv(ghostVertices.data()+rpos, ghostSizes[i], MPI_GRAPH_TYPE, i, ColoringDataTag, MPI_COMM_WORLD, &rreqs[i]);
#ifdef DEBUG_COLORING
			std::cout << " me: " << me << " to " << i << " size: " << ssizes[i] << std::endl;		
#endif      			
			std::copy(sendRemoteVertices[i].begin(), sendRemoteVertices[i].end(), sRemoteVertices.data()+spos);
      			MPI_Isend(sRemoteVertices.data()+spos, ssizes[i], MPI_GRAPH_TYPE, i, ColoringDataTag, MPI_COMM_WORLD, &sreqs[i]);
		}
		else {
      			rreqs[i] = MPI_REQUEST_NULL;
     			sreqs[i] = MPI_REQUEST_NULL;
  			}
		
	rpos+=ghostSizes[i];
	spos+=ssizes[i];
	}        

  	MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  	MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);

// exit(1);
#ifdef DEBUG_COLORING
        std::cout << " Me " << me << " All vector size " << ghostVertices.size() << std::endl;
#endif

			for (int j=0; j<ghostVertices.size(); j++) {

				if (ghostVertices[j] >=tnv){
#ifdef DEBUG_COLORING
				std::cout << "ERROR! Vertex " << j << " idx " << ghostVertices[j] << std::endl;
#endif
				exit(1);
				}


			}
}

void sendColoredRemoteVertices(const int me, const int nprocs, const DistGraph &dg, ColoredVertexSet &remoteColoredVertices, const ColorVector &vertexColor, const std::vector<GraphElem> &ghostVertices, const std::vector<GraphElem> &ghostSizes)
{
        const Graph &g = dg.getLocalGraph();
	const GraphElem base = dg.getBase(me); 
	const GraphElem bound = dg.getBound(me);
        const GraphElem nv = g.getNumVertices();
        const GraphElem ne = g.getNumEdges();

	std::vector<GraphElem> sendColoredRemoteVertices; 
	std::vector<GraphElem> receiveColoredRemoteVertices;
	std::vector<std::vector<GraphElem>> coloredVertexArray(nprocs);

	int pos=0;

	for (int i=0; i<nprocs; i++) {
		for (int v=0; v<ghostSizes[i]; v++) {
			GraphElem vertex=ghostVertices[pos+v]-base;
			if (vertexColor[vertex]!=-1){
				coloredVertexArray[i].push_back(vertex+base);
#ifdef DEBUG_COLORING
				std::cout << " me " << me << " to " << i << " Vertex " << vertex+base << std::endl;
#endif				
                        }
			}
		pos+=ghostSizes[i];
	}


   	//MPI Sending size of updates
	std::vector<GraphElem> ssizes(nprocs), rsizes(nprocs);
  	std::vector<MPI_Request> sreqs(nprocs), rreqs(nprocs);

  	for (int i = 0; i < nprocs; i++) {
    		ssizes[i] = coloredVertexArray[i].size();
#ifdef DEBUG_COLORING
    		std::cout << " Me " << me << " Size " << ssizes[i] << std::endl;			
#endif
                if (i != me) {
      			MPI_Irecv(&rsizes[i], 1, MPI_GRAPH_TYPE, i, ColoringSizeTag, MPI_COMM_WORLD, &rreqs[i]);
			MPI_Isend(&ssizes[i], 1, MPI_GRAPH_TYPE, i, ColoringSizeTag, MPI_COMM_WORLD, &sreqs[i]);
  		}
    		else {
			sreqs[i] = MPI_REQUEST_NULL;
      			rreqs[i] = MPI_REQUEST_NULL;
  		}

  	}

  	MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  	MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);

	GraphElem spos=0, rpos=0;
	int rsz=0, ssz=0;

	for (int i = 0; i < nprocs; i++) {
		ssz+=ssizes[i];
		rsz+=rsizes[i];
	}		

	receiveColoredRemoteVertices.resize(rsz);
        sendColoredRemoteVertices.resize(ssz);
	
#ifdef DEBUG_COLORING
	std::cout << "Receive size " << rsz << " Send size " << ssz << std::endl;
#endif
  	for (int i = 0; i < nprocs; i++) {
		if (i != me){
      			MPI_Irecv(receiveColoredRemoteVertices.data()+rpos, rsizes[i], MPI_GRAPH_TYPE, i, ColoringDataTag, MPI_COMM_WORLD, &rreqs[i]);
			
			std::copy(coloredVertexArray[i].begin(), coloredVertexArray[i].end(), sendColoredRemoteVertices.data()+spos);	
#ifdef DEBUG_COLORING
			std::cout << " Me " << me << " spos " << spos << " rpos " << rpos << std::endl;
#endif			
                        MPI_Isend(sendColoredRemoteVertices.data()+spos, ssizes[i], MPI_GRAPH_TYPE, i, ColoringDataTag, MPI_COMM_WORLD, &sreqs[i]);
	
		}else
		{
      			rreqs[i] = MPI_REQUEST_NULL;
  			sreqs[i] = MPI_REQUEST_NULL;
  
		}
  		rpos+=rsizes[i];
		spos+=ssizes[i];

	}

  	MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  	MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#ifdef DEBUG_COLORING
	std::cout << " receiveColoredRemoteVertices : size " << receiveColoredRemoteVertices.size() << std::endl;
#endif

 	for (int i=0; i< receiveColoredRemoteVertices.size(); i++) {
			remoteColoredVertices.insert(receiveColoredRemoteVertices[i]);
	}

#ifdef DEBUG_COLORING
	std::cout << " DEAD send " << std::endl;
#endif
				
}

GraphElem countUnassigned(const ColorVector &vertexColor)
{
      
#ifdef DEBUG_COLORING
	std::cout << " entered count unassigned. Size: " << vertexColor.size() << std::endl;    
#endif

	GraphElem localUnassigned=0;
	GraphElem globalUnassigned=0;

	for (int i=0; i<vertexColor.size(); i++){
		if (vertexColor[i]==-1)
			localUnassigned++; 
		
	}

	MPI_Allreduce(&localUnassigned, &globalUnassigned, 1, MPI_GRAPH_TYPE, MPI_SUM, MPI_COMM_WORLD);

#ifdef DEBUG_COLORING
	std::cout << " localUnassigned " << localUnassigned << " globalUnassigned " << globalUnassigned << std::endl;
#endif

	return globalUnassigned;  
}

GraphElem distCheckColoring(const int me, const int nprocs, const DistGraph &dg, const ColorVector &vertexColor, const std::vector<GraphElem> &ghostVertices, const std::vector<GraphElem> &ghostSizes)	
{
	const Graph &g = dg.getLocalGraph();
	const GraphElem base = dg.getBase(me); 
	const GraphElem bound = dg.getBound(me);
        const GraphElem nv = g.getNumVertices();
        const GraphElem ne = g.getNumEdges();

	std::vector<std::vector<ColorElem>> colorArray(nprocs);
	std::vector<ColorElem> sendRemoteColors;
	std::vector<ColorElem> receiveRemoteColors;
	std::vector<GraphElem> receiveRemoteVertices;
	std::map<GraphElem, ColorElem> remoteVertexColor; 

	GraphElem localConflicts=0, globalConflicts=0;	

#ifdef DEBUG_COLORING
	std::cout << " Entered dist checking " << std::endl;
#endif
	for (int v=0; v< ghostVertices.size(); v++)
	{
		sendRemoteColors.push_back(vertexColor[ghostVertices[v]-base]);
#ifdef DEBUG_COLORING
		std::cout << " Proc " << me << " Vertex " << ghostVertices[v] << " Color " << sendRemoteColors[v] << std::endl; 
#endif
	}
	
	//MPI Sending size of updates
	std::vector<GraphElem> rsizes(nprocs);
  	std::vector<MPI_Request> sreqs(nprocs), rreqs(nprocs);

  	for (int i = 0; i < nprocs; i++) {
    		if (i != me) {
      			MPI_Irecv(&rsizes[i], 1, MPI_GRAPH_TYPE, i, ColoringSizeTag, MPI_COMM_WORLD, &rreqs[i]);
    			MPI_Isend(&ghostSizes[i], 1, MPI_GRAPH_TYPE, i, ColoringSizeTag, MPI_COMM_WORLD, &sreqs[i]);
			}
			else {
      				rreqs[i] = MPI_REQUEST_NULL;
				sreqs[i] = MPI_REQUEST_NULL;
  			}
		}

  	MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  	MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);

	int rsz = 0, ssz =0;
	
	for (int i=0; i< nprocs; i++){
		rsz+=rsizes[i];
		ssz+=ghostSizes[i];
	}


	receiveRemoteVertices.resize(rsz);

	GraphElem rpos=0,spos=0;

  	for (int i = 0; i < nprocs; i++) {
		if (i != me){
      			MPI_Irecv(receiveRemoteVertices.data()+rpos, rsizes[i], MPI_GRAPH_TYPE, i, ColoringDataTag, MPI_COMM_WORLD, &rreqs[i]);
       			MPI_Isend(ghostVertices.data()+spos, ghostSizes[i], MPI_GRAPH_TYPE, i, ColoringDataTag,
			MPI_COMM_WORLD, &sreqs[i]);
		
		}else
		{
      			rreqs[i] = MPI_REQUEST_NULL;
     			sreqs[i] = MPI_REQUEST_NULL;
  		}
	
	rpos+=rsizes[i];
	spos+=ghostSizes[i];
	}

  	MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  	MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
	
	receiveRemoteColors.resize(rsz);

	rpos=0; spos=0;

  	for (int i = 0; i < nprocs; i++) {
		if (i != me){
      			MPI_Irecv(receiveRemoteColors.data()+rpos, rsizes[i], MPI_GRAPH_TYPE, i, ColoringDataTag, MPI_COMM_WORLD, &rreqs[i]);
			MPI_Isend(sendRemoteColors.data()+spos, ghostSizes[i], MPI_GRAPH_TYPE, i, ColoringDataTag, MPI_COMM_WORLD, &sreqs[i]);
    	
}else
		{
      			rreqs[i] = MPI_REQUEST_NULL;
     			sreqs[i] = MPI_REQUEST_NULL;
  		}
	rpos+=rsizes[i];
	spos+=ghostSizes[i];
	}

  	MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  	MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);

	for (int v=0; v<receiveRemoteVertices.size(); v++) {
			remoteVertexColor[receiveRemoteVertices[v]]=receiveRemoteColors[v];
	}


        for (int v=0; v<nv; v++){
		// skip if it is not colored
		if (vertexColor[v]==-1)
			continue;

		GraphElem e0, e1;
		
		g.getEdgeRangeForVertex(v, e0, e1);

		for(GraphElem k = e0; k < e1; k++ ){
		
 			const Edge &edge = g.getEdge(k);
			
			if(v+base == edge.tail ) //Ignore Self-loops
				continue;

			if (edge.tail>=base && edge.tail < bound) {
#ifdef DEBUG_COLORING
				std::cout << " Vertex " << v + base << " Color " << vertexColor[v] <<  " Local neighbor: " << edge.tail <<  " Color " << vertexColor[edge.tail-base] << std::endl; 
#endif
				if (vertexColor[edge.tail-base]==-1)
					continue;
				if (vertexColor[v]==vertexColor[edge.tail-base])
					localConflicts++;
		}
			else {
				ColorElem neighborColor=remoteVertexColor.find(edge.tail)->second;
#ifdef DEBUG_COLORING
				std::cout << " Vertex " << v + base << " Color " << vertexColor[v] <<  " Remote neighbor: " << edge.tail << " Color " << neighborColor << std::endl;
#endif
				if (neighborColor==-1)
					continue;
				
				if (vertexColor[v]==neighborColor)
					localConflicts++;
			}
		}
	}			
	
	MPI_Allreduce(&localConflicts, &globalConflicts, 1, MPI_GRAPH_TYPE, MPI_SUM, MPI_COMM_WORLD);

	std::cout << " Proc " << me << " Local Conflicts " << localConflicts << " Global conflicts " << globalConflicts << std::endl;
	
        return globalConflicts;
}

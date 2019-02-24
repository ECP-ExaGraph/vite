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

#include "rebuild.hpp"

extern std::ofstream ofs;

void createEdgeMPIType()
{
  EdgeInfo einfo;

  MPI_Aint begin, s, t, w;

  MPI_Get_address(&einfo, &begin);
  MPI_Get_address(&einfo.s, &s);
  MPI_Get_address(&einfo.t, &t);
  MPI_Get_address(&einfo.w, &w);

  int blens[] = { 1, 1, 1 };
  MPI_Aint displ[] = { s - begin, t - begin, w - begin };
  MPI_Datatype types[] = { MPI_GRAPH_TYPE, MPI_GRAPH_TYPE, MPI_GRAPH_TYPE };

  MPI_Type_create_struct(3, blens, displ, types, &edgeType);
  MPI_Type_commit(&edgeType);
} // createEdgeMPIType

void destroyEdgeMPIType()
{ MPI_Type_free(&edgeType); }

GraphElem distReNumber(int nprocs, ClusterLocalMap& lookUp, int me, DistGraph &dg, 
        const size_t &ssz, const size_t &rsz, const std::vector<GraphElem> &ssizes, 
        const std::vector<GraphElem> &rsizes, const std::vector<GraphElem> &svdata, 
        const std::vector<GraphElem> &rvdata, CommunityVector &cvect, 
        VertexCommMap &remoteComm) {

  const GraphElem base = dg.getBase(me);
  const Graph &g = dg.getLocalGraph();
  const GraphElem nv = g.getNumVertices();
  
  GraphElem lnc=0; //localNumUniqueCluster
  GraphElem gnc=0; //globalNumClusters
  GraphElem offset=0;
  
  PartArray parray(nprocs);  

  std::vector<GraphElem> sOldCghostData(ssz), rOldCghostData(rsz);
  std::vector<MPI_Request> screqs(nprocs), rcreqs(nprocs);
  
  // identifies communities of Vertices shared with other nodes - Ghost nodes setup in previous phase
  for (GraphElem i = 0; i < ssz; i++)
	sOldCghostData[i] = cvect[svdata[i]-base]; 

  GraphElem rpos=0, spos=0;

  // Recieve the vertex's communities
  for(int i =0; i<nprocs;i++){
    if(i != me && rsizes[i]!= 0)
      MPI_Irecv(rOldCghostData.data()+rpos, rsizes[i], MPI_GRAPH_TYPE,i,3,MPI_COMM_WORLD,&rcreqs[i]);
    else
      rcreqs[i] = MPI_REQUEST_NULL;
    rpos+= rsizes[i];
  }
  // Send the vertex's communties
  for(int i = 0; i <nprocs;i++){
    if(i!=me && ssizes[i]!=0)
      MPI_Isend(sOldCghostData.data()+spos, ssizes[i], MPI_GRAPH_TYPE,i,3,MPI_COMM_WORLD,&screqs[i]);
    else
      screqs[i] = MPI_REQUEST_NULL;
    spos+=ssizes[i];
  }
 
 
 // While waiting: add remote communities of local nodes to request array and build local lookup
  ClusterLocalMap::const_iterator stored;
  for(GraphElem i=0; i<nv;i++){
    GraphElem comm = cvect[i];
    int t = dg.getOwner(comm);
    if(t != me){
      parray[t].insert(comm);
    }
    else{
      stored = lookUp.find(comm);
      if(stored == lookUp.end()){
        lookUp.insert(std::make_pair(comm, lnc));
        lnc++;
      }
    }
  }
 
  MPI_Waitall(nprocs,screqs.data(),MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs,rcreqs.data(),MPI_STATUSES_IGNORE);

#ifdef DEBUG_PRINTF  
  ofs << "Received Sizes for dist Renumber"  << std::endl;
#endif

  std::vector<GraphElem> rOldCsizes(nprocs), sOldCsizes(nprocs);
  std::vector<MPI_Request> rOldCszreqs(nprocs), sOldCszreqs(nprocs);
  std::vector<GraphElem> rOldCdata, sOldCdata;  
  std::vector<MPI_Request> sOldCreqs(nprocs), rOldCreqs(nprocs);

  remoteComm.clear();
  
  // Then, add also remote communities of ghost vertices to the request array
  for ( GraphElem i=0; i<rpos; i++){
	const GraphElem comm = rOldCghostData[i];
        remoteComm.insert(std::make_pair(rvdata[i], comm));
        const int t = dg.getOwner(comm);
        if (t != me) {
	   parray[t].insert(comm);
      }
   }  

  GraphElem stcsz=0,rtcsz=0;
  
  // Receive and send the request sizes
  for(int i = 0; i < nprocs; i++){
    sOldCsizes[i] = parray[i].size();
    if(i != me){
      MPI_Irecv(&rOldCsizes[i],1,MPI_GRAPH_TYPE,i,4,MPI_COMM_WORLD,&rOldCszreqs[i]);
      MPI_Isend(&sOldCsizes[i],1,MPI_GRAPH_TYPE,i,4,MPI_COMM_WORLD,&sOldCszreqs[i]);
    }
    else{
      rOldCszreqs[i] = MPI_REQUEST_NULL;
      sOldCszreqs[i] = MPI_REQUEST_NULL;
    }
  }
  MPI_Waitall(nprocs,sOldCszreqs.data(),MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs,rOldCszreqs.data(),MPI_STATUSES_IGNORE);

#ifdef DEBUG_PRINTF  
  ofs << "Received Sizes of old communities" << std::endl;
#endif   

  // Aggregate size
  for(int i = 0; i < nprocs; i++){
    stcsz+= sOldCsizes[i];
    rtcsz+= rOldCsizes[i];
  }
  
  rOldCdata.resize(rtcsz);
  sOldCdata.resize(stcsz);
  rpos = spos = 0;

  // Copy full arrays of requested remote communities in buffers and send (one per node)
  for(int i = 0; i < nprocs; i++){
   
  if( i != me && rOldCsizes[i] != 0 )
      MPI_Irecv(rOldCdata.data()+rpos,rOldCsizes[i],MPI_GRAPH_TYPE,i,3,MPI_COMM_WORLD,&rOldCreqs[i]);
    else
      rOldCreqs[i] = MPI_REQUEST_NULL;

    if(i!=me && sOldCsizes[i] !=0){
      std::copy(parray[i].begin(),parray[i].end(),sOldCdata.data()+spos);
      MPI_Isend(sOldCdata.data()+spos,sOldCsizes[i],MPI_GRAPH_TYPE,i,3,MPI_COMM_WORLD,&sOldCreqs[i]);
    }
    else
      sOldCreqs[i] = MPI_REQUEST_NULL;

    spos+=sOldCsizes[i];
    rpos+=rOldCsizes[i];
  }
  MPI_Waitall(nprocs, sOldCreqs.data(),MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rOldCreqs.data(),MPI_STATUSES_IGNORE);

#ifdef DEBUG_PRINTF    
   ofs << "Received old C data " << std::endl;
#endif

  std::vector<GraphElem> lncs(nprocs);
  
  // check if locally owned communities request coming from other  nodes where already accounted locally, if not add
  for(GraphElem i =0; i < rtcsz; i++){
    GraphElem comm = rOldCdata[i];
    stored = lookUp.find(comm);
    assert(me == dg.getOwner(comm));
    if(stored == lookUp.end()){
      lookUp.insert(std::make_pair(comm, lnc));
      lnc++;
    }
  }

  // distribute size of each local number of unique clusters to all the nodes 
  MPI_Allgather(&lnc, 1, MPI_GRAPH_TYPE, lncs.data(), 1, MPI_GRAPH_TYPE,MPI_COMM_WORLD);

   for(int i = 0; i <nprocs; i++){
    gnc+= lncs[i];
  }
  for(int i = 0; i< me; i++){
    offset+= lncs[i];
  } 

#ifdef DEBUG_PRINTF    
 ofs << " After all_gather" << std::endl;
#endif

  // add offsets to local remapping
  for( ClusterLocalMap::iterator it = lookUp.begin(); it != lookUp.end(); it++){
   it->second += offset;
  }
 
  // Now each node knows all communities it owns and has renumbered them, adding n. of communities of previous node as offset
 
  spos = rpos= 0;
  
  std::vector<GraphElem> rNewCdata(stcsz), sNewCdata(rtcsz);
  std::vector<MPI_Request> rNewCreqs(nprocs), sNewCreqs(nprocs);
  
 // prepare the buffer of new community ids, by searching in the local map the ids of the old communities
  #pragma omp parallel for
  for(GraphElem i = 0; i < rtcsz; i++){
   sNewCdata[i]  = lookUp.find(rOldCdata[i])->second;
  }

// send the buffer with new communities (new communities have the same position of the old communities in the two buffers)
  for(int i = 0; i<nprocs;i++){
    if(i!= me && sOldCsizes[i]!=0)
      MPI_Irecv(rNewCdata.data()+rpos,sOldCsizes[i],MPI_GRAPH_TYPE,i,3,MPI_COMM_WORLD,&rNewCreqs[i]);
    else
      rNewCreqs[i]=MPI_REQUEST_NULL;
    if(i!= me && rOldCsizes[i]!=0)
      MPI_Isend(sNewCdata.data()+spos,rOldCsizes[i],MPI_GRAPH_TYPE,i,3,MPI_COMM_WORLD,&sNewCreqs[i]);
    else
      sNewCreqs[i]=MPI_REQUEST_NULL;

    rpos+=sOldCsizes[i];
    spos+=rOldCsizes[i];
  }
  MPI_Waitall(nprocs, sNewCreqs.data(),MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rNewCreqs.data(),MPI_STATUSES_IGNORE);

  // finally, add also the renumbered remote communities to the local map to rebuild the graph

  for(GraphElem i = 0; i<stcsz; i++){
    GraphElem oldRemComm = sOldCdata[i];
    GraphElem newRemComm = rNewCdata[i];
    stored = lookUp.find(oldRemComm);
    if(stored == lookUp.end())
      lookUp.insert(std::make_pair(oldRemComm,newRemComm));
  }

  // return new global number of vertices
  
  return gnc;
}

void fill_newEdgesMap(int me, NewEdgesMap &newEdgesMap, DistGraph& dg, CommunityVector &cvect, VertexCommMap &remoteComm, ClusterLocalMap &lookUp)
{
  const GraphElem base = dg.getBase(me);
  const GraphElem bound = dg.getBound(me);
  const Graph &g = dg.getLocalGraph();
  const GraphElem nv = g.getNumVertices();

  // Could parallelize here, but not do it yet

  for(GraphElem i = 0; i < nv; i++){
    GraphElem e0, e1;
    g.getEdgeRangeForVertex(i,e0,e1);

    if( newEdgesMap.end() == newEdgesMap.find(lookUp[cvect[i]]) )
      newEdgesMap.insert(std::make_pair( lookUp[cvect[i]], NewEdge()));
    
    NewEdge& toInsert = newEdgesMap[lookUp[cvect[i]]];

    NewEdge::iterator it;
    for(GraphElem j = e0; j<e1; j++){
      GraphElem com2;
      const Edge &edge = g.getEdge(j);
      if( edge.tail < base || edge.tail >= bound)
        com2 = lookUp[remoteComm[edge.tail]];
      else
        com2 = lookUp[cvect[edge.tail-base]];
      it = toInsert.find(com2);
      if(it != toInsert.end())
        it->second += edge.weight;
      else{
        toInsert.insert(std::make_pair(com2,edge.weight));
      }

    }
  }
}

void send_newEdges(int me, int nprocs, DistGraph* &dg, 
	GraphElem newGlobalNumVertices, NewEdgesMap& newEdgesMap )
{
  delete dg;

  GraphElem newLocalNumVertices = ((newGlobalNumVertices * (me + 1)) / nprocs) - ((newGlobalNumVertices * me) / nprocs); 
#ifdef DEBUG_PRINTF    
  ofs << "newLocalNumvertices: " << newLocalNumVertices << " newGlobalNumVertices " << newGlobalNumVertices << std::endl;
#endif  
  PartRanges parts(nprocs+1);

  parts[0]=0;

  for (int i=1; i<nprocs +1; i++)
	parts[i]=((newGlobalNumVertices * i) / nprocs);

  // set to zero initially
  GraphElem newGlobalNumEdges = 0;
  GraphElem newLocalNumEdges = 0;

  dg = new DistGraph(newGlobalNumVertices, newGlobalNumEdges);
  dg->createLocalGraph(newLocalNumVertices,newLocalNumEdges,&parts);
  Graph &g = dg->getLocalGraph();
  
  /****** Create Sendbuf ********/
  std::vector<EdgeVector> parray(nprocs);
  std::vector<NewEdge> localGraph(newLocalNumVertices);

  std::unordered_map<GraphElem, NewEdge>::iterator it;

  for(it = newEdgesMap.begin(); it != newEdgesMap.end(); it++){
    int belongProc = dg->getOwner((it->first));

#ifdef DEBUG_PRINTF    
    ofs << "Proc : " << me << " Vertex: " << it->first << " Belongs to: " << belongProc << std::endl;
#endif

    if(belongProc != me){ // Put not owned into send list
      EdgeVector& sendTo = parray[belongProc];
      NewEdge::iterator innerIt;
      for(innerIt = it->second.begin();innerIt != it->second.end(); innerIt++){
        EdgeInfo x;
        x.s = it->first;
        x.t = innerIt->first;
        x.w = innerIt->second;
        sendTo.push_back(x);
      }
    }
        else  // Put owned into localGraph
        localGraph[it->first - parts[me]] = it->second;
  }

  /*******  Send and receive *****/
  std::vector<GraphElem> sNewSize(nprocs), rNewSize(nprocs);
  std::vector<MPI_Request> ssreqs(nprocs),rsreqs(nprocs);
  std::vector<MPI_Request> sreqs(nprocs),rreqs(nprocs);
  EdgeVector rNewEdges;
  // Send size 
  for(int i = 0; i<nprocs;i++){
    if(i!= me){
      sNewSize[i] = parray[i].size();
      MPI_Irecv(&rNewSize[i],1, MPI_GRAPH_TYPE,i,3,MPI_COMM_WORLD,&rsreqs[i]);
      MPI_Isend(&sNewSize[i],1, MPI_GRAPH_TYPE,i,3,MPI_COMM_WORLD,&ssreqs[i]);
    }
    else{
      ssreqs[i] = MPI_REQUEST_NULL;
      rsreqs[i] = MPI_REQUEST_NULL;
    }
  }
  MPI_Waitall(nprocs,ssreqs.data(),MPI_STATUS_IGNORE);
  MPI_Waitall(nprocs,rsreqs.data(),MPI_STATUS_IGNORE);

  // Aggregate total recv 
  GraphElem comingSize=0;
  GraphElem rpos=0;
  for(int i =0 ; i<nprocs;i++){
    comingSize+= rNewSize[i];
  }
  rNewEdges.resize(comingSize);

  // Send and recieve data
  for(int i = 0; i<nprocs;i++){
    
    if(i!= me && rNewSize[i]!= 0)
      MPI_Irecv(rNewEdges.data()+rpos,rNewSize[i],edgeType,i,3,MPI_COMM_WORLD,&rreqs[i]);
    else
      rreqs[i] = MPI_REQUEST_NULL;
    if(i!=me && sNewSize[i]!=0)
      MPI_Isend(parray[i].data(),sNewSize[i],edgeType,i,3,MPI_COMM_WORLD,&sreqs[i]);
    else
      sreqs[i]=MPI_REQUEST_NULL; 
    
    rpos+=rNewSize[i];
  }

  MPI_Waitall(nprocs,sreqs.data(),MPI_STATUS_IGNORE);
  MPI_Waitall(nprocs,rreqs.data(),MPI_STATUS_IGNORE);
    
  /******  Reconstruction *******/
  GraphElem s;
  GraphElem t;
  GraphElem w;
  // Fill in the graph
  for(GraphElem i = 0; i < comingSize; i++){
    s = rNewEdges[i].s - parts[me];
    t = rNewEdges[i].t;
    w = rNewEdges[i].w;
    if( localGraph[s].find(t) == localGraph[s].end())
      localGraph[s].insert(std::make_pair(t,w));
    else
      localGraph[s][t] += w;
  }
  // Calculate the total number of edges
  
  for(GraphElem i =0; i<newLocalNumVertices; i++){
    newLocalNumEdges += localGraph[i].size();
  }
  
#ifdef DEBUG_PRINTF    
  ofs << " New Local NumEdges after graph reconstruction " << newLocalNumEdges << std::endl;
#endif

  MPI_Allreduce(&newLocalNumEdges,&newGlobalNumEdges,1,MPI_GRAPH_TYPE,MPI_SUM,MPI_COMM_WORLD);

#ifdef DEBUG_PRINTF    
  ofs << "Local New Edges:  " <<  newLocalNumEdges << " Global new Edges: " << newGlobalNumEdges << std::endl; 
#endif

  g.setNumEdges(newLocalNumEdges);  
  dg->setNumEdges(newGlobalNumEdges);
  
  // Write Vptr
  g.edgeListIndexes[0] = 0;
  for(GraphElem i =1; i <newLocalNumVertices+1 ;i++){
    g.edgeListIndexes[i] = g.edgeListIndexes[i-1] + localGraph[i-1].size();
  }
  // Write Edges
  GraphElem offset = 0;
  for(GraphElem i =0 ; i<newLocalNumVertices; i++){
    for(NewEdge::iterator it = localGraph[i].begin(); it!= localGraph[i].end(); it++){
      g.edgeList[offset].tail = it->first;
      g.edgeList[offset].weight = it->second;
      offset++;
          
    }
  }
  assert(offset == newLocalNumEdges);
}

void distbuildNextLevelGraph(int nprocs, int me, DistGraph*& dg, 
        const size_t &ssz, const size_t &rsz, const std::vector<GraphElem> &ssizes, 
        const std::vector<GraphElem> &rsizes, const std::vector<GraphElem> &svdata, 
        const std::vector<GraphElem> &rvdata, CommunityVector &cvect) {

  ClusterLocalMap lookUp;

  GraphElem newGlobalNumVertices;
  VertexCommMap remoteComm;  
  NewEdgesMap newEdgesMap;
  double t0, t1;
  t0 = MPI_Wtime();

  // Step 1 aggregate the alive communities
  newGlobalNumVertices = distReNumber(nprocs, lookUp, me, *dg, 
          ssz, rsz, ssizes, rsizes, svdata, rvdata, cvect, remoteComm);

  // Step 2 build Local Mapper for redistribution
  fill_newEdgesMap(me, newEdgesMap, *dg, cvect, remoteComm, lookUp);
  
  // Step 3 set up the divider and send the data for new graph
  send_newEdges(me, nprocs, dg, newGlobalNumVertices, newEdgesMap);

  t1 = MPI_Wtime();
}

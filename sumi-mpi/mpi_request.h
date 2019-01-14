/**
Copyright 2009-2018 National Technology and Engineering Solutions of Sandia, 
LLC (NTESS).  Under the terms of Contract DE-NA-0003525, the U.S.  Government 
retains certain rights in this software.

Sandia National Laboratories is a multimission laboratory managed and operated
by National Technology and Engineering Solutions of Sandia, LLC., a wholly 
owned subsidiary of Honeywell International, Inc., for the U.S. Department of 
Energy's National Nuclear Security Administration under contract DE-NA0003525.

Copyright (c) 2009-2018, NTESS

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Questions? Contact sst-macro-help@sandia.gov
*/

#ifndef SSTMAC_SOFTWARE_LIBRARIES_MPI_MPIREQUEST_H_INCLUDED
#define SSTMAC_SOFTWARE_LIBRARIES_MPI_MPIREQUEST_H_INCLUDED

#include <sstmac/software/process/key.h>
#include <sumi/collective.h>
#include <sumi-mpi/mpi_status.h>
#include <sumi-mpi/mpi_message.h>
#include <sumi-mpi/mpi_comm/mpi_comm_fwd.h>
#include <sstmac/common/sstmac_config.h>


namespace sumi {

/**
 * Persistent send operations (send, bsend, rsend, ssend)
 */
class PersistentOp
{
 public:
  /// The arguments.
  int count;
  MPI_Datatype datatype;
  MPI_Comm comm;
  int partner;
  int tag;
  void* content;
};

struct CollectiveOpBase
{

  bool packed_send;
  bool packed_recv;
  void* sendbuf;
  void* recvbuf;
  void* tmp_sendbuf;
  void* tmp_recvbuf;
  int tag;
  MPI_Op op;
  MpiType* sendtype;
  MpiType* recvtype;
  Collective::type_t ty;
  MpiComm* comm;
  int sendcnt;
  int recvcnt;
  int root;
  bool complete;

  virtual ~CollectiveOpBase(){}

 protected:
  CollectiveOpBase(MpiComm* cm);

};

struct CollectiveOp :
  public CollectiveOpBase,
  public sprockit::thread_safe_new<CollectiveOp>
{
  CollectiveOp(int count, MpiComm* comm);
  CollectiveOp(int sendcnt, int recvcnt, MpiComm* comm);


};

struct CollectivevOp :
  public CollectiveOpBase,
  public sprockit::thread_safe_new<CollectivevOp>
{
  CollectivevOp(int scnt, int* recvcnts, int* disps, MpiComm* comm);
  CollectivevOp(int* sendcnts, int* disps, int rcnt, MpiComm* comm);
  CollectivevOp(int* sendcnts, int* sdisps,
                 int* recvcnts, int* rdisps, MpiComm* comm);

  int* recvcounts;
  int* sendcounts;
  int* sdisps;
  int* rdisps;
  int size;
};

class MpiRequest :
  public sprockit::thread_safe_new<MpiRequest>
{
 public:
  typedef enum {
    Send,
    Recv,
    Collective,
    Probe
  } op_type_t;

  MpiRequest(op_type_t ty) :
   complete_(false),
   cancelled_(false),
   optype_(ty),
   persistent_op_(nullptr),
   collective_op_(nullptr)
  {
  }

  std::string toString() const {
    return "mpirequest";
  }

  std::string typeStr() const;

  static MpiRequest* construct(op_type_t ty){
    return new MpiRequest(ty);
  }

  ~MpiRequest();

  void complete(MpiMessage* msg);

  bool isComplete() const {
    return complete_;
  }

  void cancel() {
    cancelled_ = true;
    complete();
  }

  void complete() {
    complete_ = true;
  }

  void setComplete(bool flag){
    complete_ = flag;
  }

  void setPersistent(PersistentOp* op) {
    persistent_op_ = op;
  }

  PersistentOp* persistentData() const {
    return persistent_op_;
  }

  void setCollective(CollectiveOpBase* op) {
    collective_op_ = op;
  }

  CollectiveOpBase* collectiveData() const {
    return collective_op_;
  }

  const MPI_Status& status() const {
    return stat_;
  }

  bool isCancelled() const {
    return cancelled_;
  }

  bool isPersistent() const {
    return persistent_op_;
  }

  bool isCollective() const {
    return collective_op_;
  }

  op_type_t optype() const {
    return optype_;
  }

 private:
  MPI_Status stat_;
  bool complete_;
  bool cancelled_;
  op_type_t optype_;

  PersistentOp* persistent_op_;
  CollectiveOpBase* collective_op_;

};

}

#endif

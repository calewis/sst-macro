/**
Copyright 2009-2017 National Technology and Engineering Solutions of Sandia, 
LLC (NTESS).  Under the terms of Contract DE-NA-0003525, the U.S.  Government 
retains certain rights in this software.

Sandia National Laboratories is a multimission laboratory managed and operated
by National Technology and Engineering Solutions of Sandia, LLC., a wholly 
owned subsidiary of Honeywell International, Inc., for the U.S. Department of 
Energy's National Nuclear Security Administration under contract DE-NA0003525.

Copyright (c) 2009-2017, NTESS

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * Neither the name of Sandia Corporation nor the names of its
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

#include <gni/gni_transport.h>
#include <cstring>

namespace sumi {

void
gni_transport::rdma_put_done(const message::ptr& msg)
{
  if (msg->needs_recv_ack() && !msg->has_transaction_id()){
    gni_debug("Rank %d software acking RDMA put to %d",
        rank_, msg->recver());
    smsg_send(msg->recver(), msg, RDMA_PUT_RECV_ACK);
  }
  if (msg->needs_send_ack()){
    message::ptr cln = msg->clone_ack();
    handle(cln);
  }
}

void
gni_transport::do_rdma_put(int dst, const message::ptr &msg)
{
  public_buffer& send_buf = msg->local_buffer();
  public_buffer& recv_buf = msg->remote_buffer();

  if (msg->byte_length() == 0){
    rdma_put_done(msg);
    return;
  }
  else if (dst == rank_){
    ::memcpy(recv_buf.ptr, send_buf.ptr, msg->byte_length());
    rdma_put_done(msg);
    return;
  }

  int tag = allocate_rdma_tag(msg);

  gni_debug("Rank %d RDMA put buffer %p into buffer %p on dst %d on tag %d",  
    rank_, (void*) send_buf, (void*) recv_buf, dst, tag);

  post_rdma(
    dst,
    msg->byte_length(),
    tag,
    send_buf.ptr,
    send_buf.mem_handle,
    recv_buf.ptr,
    recv_buf.mem_handle,
    GNI_POST_RDMA_PUT,
    GNI_CQMODE_GLOBAL_EVENT,
    //add any metadata for remote event acks
    PUT_ACK,
    msg->transaction_id());

  rdma_messages_[tag] = msg;
}

}
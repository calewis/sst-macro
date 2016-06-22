#include <sstmac/hardware/packet_flow/packet_flow_packetizer.h>

/*
 *  This file is part of SST/macroscale:
 *               The macroscale architecture simulator from the SST suite.
 *  Copyright (c) 2009 Sandia Corporation.
 *  This software is distributed under the BSD License.
 *  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
 *  the U.S. Government retains certain rights in this software.
 *  For more information, see the LICENSE file in the top
 *  SST/macroscale directory.
 */

#include <sstmac/hardware/topology/structured_topology.h>
#include <sstmac/hardware/network/network_message.h>
#include <sstmac/hardware/packet_flow/packet_flow_packetizer.h>
#include <sstmac/hardware/node/node.h>
#include <sstmac/software/process/operating_system.h>
#include <sstmac/common/event_manager.h>
#include <sstmac/common/event_callback.h>
#include <sstmac/common/stats/stat_spyplot.h>
#include <sprockit/errors.h>
#include <sprockit/util.h>
#include <sprockit/sim_parameters.h>
#include <sprockit/keyword_registration.h>

#include <stddef.h>

RegisterNamespaces("congestion_delays", "congestion_matrix");

namespace sstmac {
namespace hw {

SpktRegister("cut_through | null", packetizer, packet_flow_cut_through_packetizer);
SpktRegister("simple", packetizer, packet_flow_simple_packetizer);

packet_flow_nic_packetizer::packet_flow_nic_packetizer() :
 inj_buffer_(0),
 ej_buffer_(0),
 congestion_spyplot_(0),
 congestion_hist_(0)
{
}

void
packet_flow_nic_packetizer::set_event_parent(event_scheduler *m)
{
  packetizer::set_event_parent(m);
  inj_buffer_->set_event_parent(m);
  ej_buffer_->set_event_parent(m);
#if !SSTMAC_INTEGRATED_SST_CORE
  if (congestion_hist_) m->register_stat(congestion_hist_);
  if (congestion_spyplot_) m->register_stat(congestion_spyplot_);
#endif
}

void
packet_flow_nic_packetizer::init_factory_params(sprockit::sim_parameters *params)
{
  packet_flow_packetizer::init_factory_params(params);

  my_addr_ = node_id(params->get_int_param("id"));
  init_loc_id(event_loc_id(my_addr_));

  acc_delay_ = params->get_optional_bool_param("accumulate_congestion_delay",false);

  if (params->has_namespace("congestion_delay_histogram")){
    sprockit::sim_parameters* congestion_params = params->get_namespace("congestion_delay_histogram");
    congestion_hist_ = test_cast(stat_histogram, stat_collector_factory::get_optional_param("type", "histogram", congestion_params));
    if (!congestion_hist_){
      spkt_throw_printf(sprockit::value_error,
        "congestion delay stats must be histogram, %s given",
        congestion_params->get_param("type").c_str());
    }
  }

  if (params->has_namespace("congestion_delay_matrix")){
    sprockit::sim_parameters* congestion_params = params->get_namespace("congestion_delay_matrix");
    congestion_spyplot_ = test_cast(stat_spyplot, stat_collector_factory::get_optional_param("type", "spyplot_png", congestion_params));
    if (!congestion_spyplot_){
      spkt_throw_printf(sprockit::value_error,
        "congestion matrix stats must be spyplot or spyplot_png, %s given",
        congestion_params->get_param("type").c_str());
    }
  }
  inj_bw_ = params->get_bandwidth_param("injection_bandwidth");
  if (params->has_param("ejection_bandwidth")){
    ej_bw_ = params->get_bandwidth_param("ejection_bandwidth");
  } else {
    ej_bw_ = inj_bw_;
  }

  buffer_size_ = params->get_optional_byte_length_param("eject_buffer_size", 1<<30);

  int one_vc = 1;
  //total hack for now, assume that the buffer itself has a low latency link to the switch
  timestamp small_latency(10e-9);
  packet_flow_bandwidth_arbitrator* inj_arb = packet_flow_bandwidth_arbitrator_factory::get_optional_param(
        "arbitrator", "cut_through", params);
  inj_arb->set_outgoing_bw(inj_bw_);
  inj_buffer_ = new packet_flow_injection_buffer(small_latency, inj_arb, packetSize());

  //total hack for now, assume that the buffer has a delayed send, but ultra-fast credit latency
  packet_flow_bandwidth_arbitrator* ej_arb = packet_flow_bandwidth_arbitrator_factory::get_optional_param(
        "arbitrator", "cut_through", params);
  ej_arb->set_outgoing_bw(ej_bw_);
  timestamp inj_lat = params->get_time_param("injection_latency");
  ej_buffer_ = new packet_flow_eject_buffer(inj_lat, small_latency, buffer_size_, ej_arb);
}

void
packet_flow_nic_packetizer::set_acker(event_handler *handler)
{
  inj_buffer_->set_acker(handler);
}

//
// Goodbye.
//
packet_flow_nic_packetizer::~packet_flow_nic_packetizer() throw ()
{
  if (inj_buffer_) delete inj_buffer_;
  if (ej_buffer_) delete ej_buffer_;
}

void
packet_flow_nic_packetizer::finalize_init()
{
  inj_buffer_->set_event_location(my_addr_);
  ej_buffer_->set_event_location(my_addr_);

  inj_buffer_->set_accumulate_delay(acc_delay_);
  ej_buffer_->set_accumulate_delay(acc_delay_);

  packetizer::finalize_init();
}

void
packet_flow_packetizer::handle(event *ev)
{
  if (ev->is_credit()){
    recv_credit(static_cast<packet_flow_credit*>(ev));
  } else {
    recv_packet(static_cast<packet_flow_payload*>(ev));
  }
}

void
packet_flow_nic_packetizer::recv_credit(packet_flow_credit* credit)
{
  inj_buffer_->handle_credit(credit);
  int vn = 0;
  sendWhatYouCan(vn);
}

bool
packet_flow_nic_packetizer::spaceToSend(int vn, int num_bits) const
{
  //convert back to bytes
  return inj_buffer_->space_to_send(num_bits/8);
}

void
packet_flow_nic_packetizer::inject(int vn, long bytes, long byte_offset, message* msg)
{
  packet_flow_payload* payload = new packet_flow_payload(msg, bytes, byte_offset);
  inj_buffer_->handle_payload(payload);
}

void
packet_flow_nic_packetizer::recv_packet_common(packet_flow_payload* pkt)
{
  ej_buffer_->return_credit(pkt);

  if (congestion_hist_){
    congestion_hist_->collect(pkt->delay_us()*1e-6); //convert to seconds
  }
  if (congestion_spyplot_){
    long delay_ns = pkt->delay_us() * 1e3; //go to ns
    congestion_spyplot_->add(pkt->fromaddr(), pkt->toaddr(), delay_ns);
  }
}

void
packet_flow_nic_packetizer::set_output(int inj_port, connectable* sw, int credits)
{
  event_handler* handler = safe_cast(event_handler, sw);
  inj_buffer_->set_output(0, inj_port, handler);
  inj_buffer_->init_credits(0, credits);
}

void
packet_flow_nic_packetizer::set_input(int ej_port, connectable* sw)
{
  event_handler* handler = safe_cast(event_handler, sw);
  int only_port = 0;
  ej_buffer_->set_output(only_port, only_port, this);
  ej_buffer_->set_input(only_port, ej_port, handler);
}

void
packet_flow_simple_packetizer::recv_packet(packet_flow_payload *pkt)
{
  recv_packet_common(pkt);
  int vn = 0;
  packetArrived(vn, pkt);
}

void
packet_flow_cut_through_packetizer::recv_packet(packet_flow_payload *pkt)
{
  int vn = 0;
  recv_packet_common(pkt);
  timestamp delay(pkt->num_bytes() / pkt->bw());
  debug_printf(sprockit::dbg::packet_flow,
    "packet %s scheduled to arrive at packetizer after delay of t=%12.6es",
     pkt->to_string().c_str(), delay.sec());
  send_delayed_self_event_queue(delay,
    new_event(this, &packetizer::packetArrived, vn, pkt));
}

}
} // end of namespace sstmac.




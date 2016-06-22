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

#include <sstmac/hardware/switch/network_switch.h>
#include <sstmac/hardware/packet_flow/packet_flow_switch.h>
#include <sstmac/hardware/packet_flow/packet_flow_stats.h>
#include <sstmac/hardware/nic/nic.h>
#include <sstmac/hardware/switch/dist_dummyswitch.h>
#include <sstmac/common/event_manager.h>
#include <sstmac/common/stats/stat_spyplot.h>
#include <sstmac/common/stats/stat_global_int.h>
#include <sprockit/util.h>
#include <sprockit/sim_parameters.h>
#include <sprockit/keyword_registration.h>

RegisterNamespaces("congestion_stats");

namespace sstmac {
namespace hw {

#if !SSTMAC_INTEGRATED_SST_CORE
SpktRegister("packet_flow", network_switch, packet_flow_switch);
#endif

ImplementIntegratedComponent(packet_flow_switch);

template <class T>
void
set_ev_parent(T& themap, event_scheduler* m)
{
  typename T::iterator it, end = themap.end();
  for (it=themap.begin(); it != end; ++it) {
    it->second->set_event_parent(m);
  }
}

template <class T>
void
vec_set_ev_parent(std::vector<T*>& themap, event_scheduler* m)
{
  for (int i=0; i < themap.size(); ++i){
    T* t = themap[i];
    if (t) t->set_event_parent(m);
  }
}

packet_flow_params::~packet_flow_params()
{
  delete link_arbitrator_template;
}


void
packet_flow_abstract_switch::init_factory_params(sprockit::sim_parameters *params)
{
  network_switch::init_factory_params(params);
  params_ = new packet_flow_params;
  params_->link_bw =
    params->get_bandwidth_param("link_bandwidth");
  params_->hop_lat =
    params->get_time_param("hop_latency");
  params_->xbar_output_buffer_num_bytes
    = params->get_byte_length_param("output_buffer_size");
  params_->crossbar_bw =
    params->get_bandwidth_param("crossbar_bandwidth");
  params_->xbar_input_buffer_num_bytes
    = params->get_byte_length_param("input_buffer_size");

  if (params->has_param("ejection_bandwidth")){
    params_->ej_bw =
       params->get_bandwidth_param("ejection_bandwidth");
  } else {
    params_->ej_bw =
       params->get_bandwidth_param("injection_bandwidth");
  }

  params_->link_arbitrator_template
    = packet_flow_bandwidth_arbitrator_factory::get_optional_param(
        "arbitrator", "cut_through", params);

  /**
    sstkeyword {
      docstring=Enables output queue depth reporting.ENDL
      If set to true, warnings will be provided each time an output queue increases by a given number.
      This can only be enabled if sanity check is enabled by configure.;
    }
  */
  params_->queue_depth_reporting =
      params->get_optional_bool_param("sanity_check_queue_depth_reporting",false);

  /**
    sstkeyword {
      docstring=Sets the count delta for output queue depth reporting.ENDL
      The default is 100.;
    }
  */
  params_->queue_depth_delta =
      params->get_optional_int_param("sanity_check_queue_depth_delta", 100);
}

#if SSTMAC_INTEGRATED_SST_CORE
packet_flow_switch::packet_flow_switch(
  SST::ComponentId_t id,
  SST::Params& params
) : packet_flow_abstract_switch(id, params),
  congestion_spyplot_(0),
  bytes_sent_(0),
  byte_hops_(0),
  xbar_(0)
{
  init_factory_params(SSTIntegratedComponent::params_);
  init_sst_params(params);
}
#else
packet_flow_switch::packet_flow_switch() :
 xbar_(0),
 congestion_spyplot_(0),
 bytes_sent_(0),
 byte_hops_(0)
{
}
#endif

packet_flow_switch::~packet_flow_switch()
{
  if (xbar_) delete xbar_;
  if (bytes_sent_) delete bytes_sent_;
  if (byte_hops_) delete byte_hops_;
  if (congestion_spyplot_) delete congestion_spyplot_;
 
  int nbuffers = out_buffers_.size();
  for (int i=0; i < nbuffers; ++i){
    packet_flow_sender* buf = out_buffers_[i];
    if (buf) delete buf;
  }

  if (params_) delete params_;
}


void
packet_flow_switch::init_factory_params(sprockit::sim_parameters *params)
{
  packet_flow_abstract_switch::init_factory_params(params);

  acc_delay_ = params->get_optional_bool_param("accumulate_congestion_delay",false);

  packet_size_ = params->get_optional_byte_length_param("mtu", 4096);

  if (params->has_namespace("congestion_matrix")){
    sprockit::sim_parameters* congestion_params = params->get_namespace("congestion_matrix");
    congestion_spyplot_ = test_cast(stat_spyplot, stat_collector_factory::get_optional_param("type", "spyplot_png", congestion_params));
    if (!congestion_spyplot_){
      spkt_throw_printf(sprockit::value_error,
        "packet flow congestion stats must be spyplot or spyplot_png, %s given",
        congestion_params->get_param("type").c_str());
    }
  }

  if (params->has_namespace("bytes_sent")){
    sprockit::sim_parameters* byte_params = params->get_namespace("bytes_sent");
    bytes_sent_ = test_cast(stat_bytes_sent, stat_collector_factory::get_optional_param("type", "bytes_sent", byte_params));
    if (!bytes_sent_){
      spkt_throw_printf(sprockit::value_error,
        "packet flow bytes sent stats must be bytes_sent, %s given",
        byte_params->get_param("type").c_str());
    }
    bytes_sent_->set_id(my_addr_);
  }

  if (params->has_namespace("byte_hops")) {
    sprockit::sim_parameters* traffic_params = params->get_namespace("byte_hops");
    byte_hops_ = test_cast(stat_global_int, stat_collector_factory::get_optional_param("type", "global_int", traffic_params));
    byte_hops_->set_label("Byte Hops");
  }
}

void
packet_flow_switch::set_topology(topology *top)
{
  if (bytes_sent_) bytes_sent_->set_topology(top);
  network_switch::set_topology(top);
}

void
packet_flow_switch::initialize()
{
  xbar_->set_accumulate_delay(acc_delay_);
  int nbuffers = out_buffers_.size();
  int buffer_inport = 0;
  for (int i=0; i < nbuffers; ++i){
    int outport = i;
    packet_flow_sender* buf = out_buffers_[outport];
    if (buf){ //might be sparse
      xbar_->set_output(outport, buffer_inport, buf);
      xbar_->init_credits(outport, buf->num_initial_credits());
      buf->set_input(buffer_inport, outport, xbar_);
      buf->set_event_location(my_addr_);
      buf->set_accumulate_delay(acc_delay_);
    }
  }
}

packet_flow_crossbar*
packet_flow_switch::crossbar(config* cfg)
{
  if (!xbar_) {
    double xbar_bw = params_->crossbar_bw;
    if (cfg->ty == WeightedConnection){
      xbar_bw *= cfg->xbar_weight;
    }
    debug_printf(sprockit::dbg::packet_flow | sprockit::dbg::packet_flow_config,
      "Switch %d: creating crossbar with bandwidth %12.8e",
      int(my_addr_), xbar_bw);
    xbar_ = new packet_flow_crossbar(
              timestamp(0), //assume zero-time send
              params_->hop_lat, //delayed credits
              params_->crossbar_bw,
              router_->max_num_vc(),
              params_->xbar_input_buffer_num_bytes,
              params_->link_arbitrator_template->clone(-1/*fake bw*/));
    xbar_->configure_basic_ports(topol()->max_num_ports());
    xbar_->set_event_location(my_addr_);
  }
  return xbar_;
}

void
packet_flow_switch::resize_buffers()
{
  if (out_buffers_.empty()) out_buffers_.resize(top_->max_num_ports());
}

packet_flow_sender*
packet_flow_switch::output_buffer(int port, config* cfg)
{
  if (!out_buffers_[port]){
    bool inj_port = top_->is_injection_port(port);
    double total_link_bw = inj_port ? params_->ej_bw : params_->link_bw;
    int dst_buffer_size = params_->xbar_input_buffer_num_bytes;
    int src_buffer_size = params_->xbar_output_buffer_num_bytes;
    timestamp lat = params_->hop_lat;
    switch(cfg->ty){
      case BasicConnection:
        break;
      case RedundantConnection:
        total_link_bw *= cfg->red;
        src_buffer_size *= cfg->red;
        break;
       case WeightedConnection:
        total_link_bw *= cfg->link_weight;
        src_buffer_size *= cfg->src_buffer_weight;
        dst_buffer_size *= cfg->dst_buffer_weight;
        break;
      case FixedBandwidthConnection:
        total_link_bw = cfg->bw;
        break;
      case FixedConnection:
        total_link_bw = cfg->bw;
        lat = cfg->latency;
        break;
      default:
        spkt_throw_printf(sprockit::value_error,
          "bad connection::config enum %d", cfg->ty);
    }

    debug_printf(sprockit::dbg::packet_flow | sprockit::dbg::packet_flow_config,
      "Switch %d: making buffer with bw=%10.6e on port=%d with buffer size %d going into buffer size %d",
      int(my_addr_), total_link_bw, port, src_buffer_size, dst_buffer_size);

    packet_flow_network_buffer* out_buffer
      = new packet_flow_network_buffer(
                  params_->hop_lat,
                  timestamp(0), //assume credit latency to xbar is free
                  src_buffer_size,
                  router_->max_num_vc(),
                  packet_size_,
                  params_->link_arbitrator_template->clone(total_link_bw));

    out_buffer->set_event_location(my_addr_);
    int buffer_outport = 0;
    out_buffer->init_credits(buffer_outport, dst_buffer_size);
    out_buffer->set_sanity_params(params_->queue_depth_reporting,
                                params_->queue_depth_delta);
    out_buffers_[port] = out_buffer;
  }
  return out_buffers_[port];
}

void
packet_flow_switch::connect_output(
  int src_outport,
  int dst_inport,
  connectable* mod,
  config* cfg)
{
  resize_buffers();

  //create an output buffer for the port
  packet_flow_sender* out_buffer = output_buffer(src_outport, cfg);
  out_buffer->set_output(src_outport, dst_inport, safe_cast(event_handler, mod));
}

void
packet_flow_switch::connect_input(
  int src_outport,
  int dst_inport,
  connectable* mod,
  config* cfg)
{
  crossbar(cfg)->set_input(dst_inport, src_outport, safe_cast(event_handler, mod));
}

void
packet_flow_switch::connect(
  int src_outport,
  int dst_inport,
  connection_type_t ty,
  connectable* mod,
  config* cfg)
{
  switch(ty) {
    case output:
      connect_output(src_outport, dst_inport, mod, cfg);
      break;
    case input:
      connect_input(src_outport, dst_inport, mod, cfg);
      break;
  }
}

void
packet_flow_switch::connect_injector(int src_outport, int dst_inport, event_handler* nic)
{
  connectable* inp = safe_cast(connectable, nic);
  connect_input(src_outport, dst_inport, inp, NULL); //no cfg
}

void
packet_flow_switch::connect_ejector(int src_outport, int dst_inport, event_handler* nic)
{
  connectable* inp = safe_cast(connectable, nic);
  connect_output(src_outport, dst_inport, inp, NULL); //no cfg
}

std::vector<switch_id>
packet_flow_switch::connected_switches() const
{
  std::vector<switch_id> ret;
  ret.reserve(out_buffers_.size());
  int idx = 0;
  for (int b=0; b < out_buffers_.size(); ++b){
    packet_flow_buffer* buf = test_cast(packet_flow_buffer, out_buffers_[b]);
    if (buf) ret[idx++] = buf->output_location().convert_to_switch_id();
  }
  return ret;
}

void
packet_flow_switch::deadlock_check()
{
  xbar_->deadlock_check();
}

void
packet_flow_switch::set_event_manager(event_manager* m)
{
  network_switch::set_event_manager(m);
  if (!xbar_){
    spkt_throw(sprockit::value_error,
       "crossbar uninitialized on switch");
  }
#if !SSTMAC_INTEGRATED_SST_CORE
  if (congestion_spyplot_){
    xbar_->set_congestion_spyplot(congestion_spyplot_);
    for (int i=0; i < out_buffers_.size(); ++i){
      packet_flow_buffer* buf = test_cast(packet_flow_buffer, out_buffers_[i]);
      if (buf) buf->set_congestion_spyplot(congestion_spyplot_);
    }
    m->register_stat(congestion_spyplot_);
  }

  if (bytes_sent_){
    xbar_->set_bytes_sent_collector(bytes_sent_);
    m->register_stat(bytes_sent_);
  }

  if (byte_hops_) {
    xbar_->set_byte_hops_collector(byte_hops_);
    m->register_stat(byte_hops_);
  }
#endif
  vec_set_ev_parent(out_buffers_, this);
  xbar_->set_event_parent(this);
}

int
packet_flow_switch::queue_length(int port) const
{
  packet_flow_buffer* buf = static_cast<packet_flow_buffer*>(out_buffers_[port]);
  return buf->queue_length();
}

void
packet_flow_switch::handle(event* ev)
{
  //this should only happen in parallel mode...
  //this means we are getting a message that has crossed the parallel boundary
  packet_flow_interface* fpack = interface_cast(packet_flow_interface, ev);
  switch (fpack->type()) {
    case packet_flow_interface::credit: {
      packet_flow_credit* credit = static_cast<packet_flow_credit*>(fpack);
      out_buffers_[credit->port()]->handle_credit(credit);
      break;
    }
    case packet_flow_interface::payload: {
      packet_flow_payload* payload = static_cast<packet_flow_payload*>(fpack);
      router_->route(payload);
      xbar_->handle_payload(payload);
      break;
    }
  }
}

std::string
packet_flow_switch::to_string() const
{
  return sprockit::printf("packet flow switch %d", int(my_addr_));
}

}
}




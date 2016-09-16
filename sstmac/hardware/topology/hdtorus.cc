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

// hdtorus.cc: Implementation of a high dimensional torus network.
//
// Author: Jeremiah Wilke <jjwilke@sandia.gov>
//

#include <sstmac/hardware/topology/hdtorus.h>
#include <stdio.h>
#include <sstream>
#include <sprockit/stl_string.h>
#include <sprockit/output.h>
#include <sprockit/util.h>
#include <sprockit/sim_parameters.h>

namespace sstmac {
namespace hw {

SpktRegister("torus | hdtorus", topology, hdtorus,
            "hdtorus implements a high-dimension torus with an arbitrary number of dimensions");

static bool
equals(const std::vector<int>& coords, int x, int y, int z)
{
  if (coords.size() != 3) {
    return false;
  }
  return coords[0] == x && coords[1] == y && coords[2] == z;
}

hdtorus::hdtorus(sprockit::sim_parameters* params) :
  cartesian_topology(params,
                     InitMaxPortsIntra::I_Remembered,
                     InitGeomEjectID::I_Remembered)
{
  num_switches_ = 1;
  diameter_ = 0;
  for (int i = 0; i < (int) dimensions_.size(); i++) {
    num_switches_ *= dimensions_[i];
    diameter_ += dimensions_[i] / 2;
  }

  max_ports_intra_network_ = 2 * dimensions_.size();
  eject_geometric_id_ = max_ports_intra_network_;
}

coordinates
hdtorus::neighbor_at_port(switch_id sid, int port)
{
  coordinates my_coords = switch_coords(sid);
  if (port < 0){
    return my_coords;
  }

  int dir = port % 2;
  int dim = port / 2;

  int adder = 1;
  if (dir == 1) //negative
    adder = -1;

  my_coords[dim] += adder;
  if (my_coords[dim] < 0)
    my_coords[dim] = dimensions_[dim];
  return my_coords;
}

void
hdtorus::compute_switch_coords(switch_id uid, coordinates& coords) const
{
  long div = 1;
  int ndim = dimensions_.size();
  for (int i = 0; i < ndim; i++) {
    coords[i] = (uid / div) % dimensions_[i];
    div *= dimensions_[i];
  }
}

switch_id
hdtorus::switch_number(const coordinates& coords) const
{
  long ret = 0;
  long mult = 1;
  for (int i = 0; i < (int) dimensions_.size(); i++) {
    if (coords[i] >= dimensions_[i]) {
      spkt_throw_printf(sprockit::illformed_error,
                       "hdtorus::get_switch_id: dimension %d coordinate %d exceeds maximum %d\n"
                       "check that topology_geometry matches map file",
                       i, coords[i], dimensions_[i] - 1);
    }
    ret += coords[i] * mult;
    mult *= dimensions_[i];
  }
  return switch_id(ret);
}

int
hdtorus::distance(int dim, int dir, int src, int dst) const
{
  if (dir == pos) {
    return ((dst - src) + dimensions_[dim]) % dimensions_[dim];
  }
  else {
    return ((src - dst) + dimensions_[dim]) % dimensions_[dim];
  }
}

int
hdtorus::shortest_distance(int dim, int src, int dst) const
{
  int up_distance, down_distance;
  if (dst > src) {
    up_distance = dst - src;
    down_distance = src + (dimensions_[dim] - dst);
  }
  else {
    up_distance = dst + (dimensions_[dim] - src);
    down_distance = src - dst;
  }

  if (up_distance > down_distance) {
    //shorter to go down
    return ((src - dst) + dimensions_[dim]) % dimensions_[dim];
  }
  else {
    //shorter to go up
    return ((dst - src) + dimensions_[dim]) % dimensions_[dim];
  }
}

bool
hdtorus::shortest_path_positive(
  int dim,
  const coordinates &src_coords,
  const coordinates &dst_coords) const
{
  int up_distance, down_distance;
  int src = src_coords[dim];
  int dst = dst_coords[dim];
  if (dst > src) {
    up_distance = dst - src;
    down_distance = src + (dimensions_[dim] - dst);
  }
  else {
    up_distance = dst + (dimensions_[dim] - src);
    down_distance = src - dst;
  }

  return up_distance <= down_distance;
}

void
hdtorus::pick_vc(structured_routable::path& path) const
{
  if (path.metadata_bit(structured_routable::crossed_timeline)){
    path.vc = 1;
  } else {
    path.vc = 0;
  }
}

void
hdtorus::up_path(
  int dim,
  const coordinates &src_coords,
  const coordinates &dst_coords,
  structured_routable::path& path) const
{
  //shorter to go up
  //path.dir = pos;
  path.outport = convert_to_port(dim, pos);

  bool reset_dim = ((src_coords[dim] + 1) % dimensions_[dim]) == dst_coords[dim];
  bool wrapped = src_coords[dim] == (dimensions_[dim]-1);

  if (wrapped){
    path.set_metadata_bit(structured_routable::crossed_timeline);
  }

  pick_vc(path);

  if (reset_dim){
    path.unset_metadata_bit(structured_routable::crossed_timeline); //we reached this dim
  }

}

void
hdtorus::down_path(
  int dim,
  const coordinates &src_coords,
  const coordinates &dst_coords,
  structured_routable::path& path) const
{
  //shorter to go down
  //path.dir = neg;
  path.outport = convert_to_port(dim, neg);

  bool reset_dim = src_coords[dim] == ((dst_coords[dim] + 1) % dimensions_[dim]);
  bool wrapped = src_coords[dim] == 0;
  if (wrapped){
    path.set_metadata_bit(structured_routable::crossed_timeline);
  }

  pick_vc(path);

  if (reset_dim){
    path.unset_metadata_bit(structured_routable::crossed_timeline);
  }
}


void
hdtorus::productive_path(
  int dim,
  const coordinates &src_coords,
  const coordinates &dst_coords,
  structured_routable::path& path) const
{
  int up_distance, down_distance;

  int src = src_coords[dim];
  int dst = dst_coords[dim];
  if (dst > src) {
    up_distance = dst - src;
    down_distance = src + (dimensions_[dim] - dst);
  }
  else {
    up_distance = dst + (dimensions_[dim] - src);
    down_distance = src - dst;
  }


  if (shortest_path_positive(dim, src_coords, dst_coords)){
    up_path(dim, src_coords, dst_coords, path);
  }
  else {
    down_path(dim, src_coords, dst_coords, path);
  }

}

void
hdtorus::minimal_route_to_coords(
  const coordinates &src_coords,
  const coordinates &dest_coords,
  structured_routable::path& path) const
{
  //deadlock rules require + dir first
  for (int i=0; i < src_coords.size(); ++i) {
    if (src_coords[i] != dest_coords[i] && 
      shortest_path_positive(i, src_coords, dest_coords)) {
        up_path(i, src_coords, dest_coords, path);
        return;
    }
  }

  for (int i=0; i < src_coords.size(); ++i) {
    if (src_coords[i] != dest_coords[i]){
      //must be down
      down_path(i, src_coords, dest_coords, path);
      return;
    }
  }

}

int
hdtorus::minimal_distance(
  const coordinates &src_coords,
  const coordinates &dest_coords) const
{
#if SSTMAC_SANITY_CHECK
  if (src_coords.size() < dimensions_.size()) {
    spkt_throw_printf(sprockit::illformed_error,
                     "hdtorus::minimal_distance: source coordinates too small");
  }
  if (dest_coords.size() < dimensions_.size()) {
    spkt_throw_printf(sprockit::illformed_error,
                     "hdtorus::minimal_distance: dest coordinates too small");
  }
#endif
  int dist = 0;
  for (int i=0; i < dimensions_.size(); ++i) {
    dist += shortest_distance(i, src_coords[i], dest_coords[i]);
  }
  return dist;
}

void
hdtorus::connect_dim(
  sprockit::sim_parameters* params,
  int dim,
  connectable* center,
  connectable* plus_partner,
  connectable* minus_partner)
{
  int pos_outport = convert_to_port(dim, pos);
  int neg_outport = convert_to_port(dim, neg);

  sprockit::sim_parameters* pos_params = get_port_params(params, pos_outport);
  sprockit::sim_parameters* neg_params = get_port_params(params, neg_outport);

  center->connect_output(
    pos_params,
    pos_outport, neg_outport,
    plus_partner);
  plus_partner->connect_input(
    neg_params,
    pos_outport, neg_outport,
    center);

  center->connect_output(
    neg_params,
    neg_outport, pos_outport,
    minus_partner);
  minus_partner->connect_input(
    pos_params,
    neg_outport, pos_outport,
    center);
}

void
hdtorus::connect_objects(sprockit::sim_parameters* params,
                         internal_connectable_map& objects)
{
  top_debug("hdtorus: connecting %d switches",
    int(objects.size()));

  sprockit::sim_parameters* link_params = params->get_namespace("link");
  double bw = link_params->get_bandwidth_param("bandwidth");
  int bufsize = params->get_byte_length_param("buffer_size");
  int ndims = dimensions_.size();
  for (int i=0; i < ndims; ++i){
    double port_bw = bw * red_[i];
    int credits = bufsize * red_[i];
    for (int dir=0; dir < 2; ++dir){
      int port = convert_to_port(i, dir);
      setup_port_params(port, credits, port_bw, link_params, params);
    }
  }

  internal_connectable_map::iterator it;
  for (it =  objects.begin(); it !=  objects.end(); ++it) {
    switch_id me(it->first);
    coordinates coords = switch_coords(me);

    top_debug("hdtorus: connecting switch %d,%s",
        int(me), stl_string(coords).c_str());

    std::vector<int> connected;
    std::vector<int> connected_red;

    for (int z = 0; z < (int) dimensions_.size(); z++) {
      int num_this_dim = dimensions_[z];
      if (num_this_dim == 1){
        continue; //no connections to be done
      }

      coordinates connect1 = coords;
      coordinates connect2 = coords;
      bool wrap_pos = false;
      bool wrap_neg = false;
      if (coords[z] == 0) {
        connect1[z]++;
        connect2[z] = dimensions_[z] - 1;
        wrap_neg = true;
      }
      else if (coords[z] == (dimensions_[z] - 1)) {
        connect1[z] = 0;
        connect2[z]--;
        wrap_pos = true;
      }
      else {
        connect1[z]++;
        connect2[z]--;
      }
      switch_id addr1 = switch_number(connect1);
      switch_id addr2 = switch_number(connect2);

      connectable* dest1 = NULL, *dest2 = NULL;

      /** We use a special lookup function rather than directly
          getting from the map.  The object might not exist
          due to parallel partitioning */
      dest1 = objects[addr1];
      dest2 = objects[addr2];



      top_debug("switch %d,%s for dim %d connecting in the pos dir to %d,%s : %s",
         int(me), stl_string(coords).c_str(), z,
         int(addr1), stl_string(connect1).c_str(),
         dest1->to_string().c_str());

      top_debug("switch %d,%s for dim %d connecting in the neg dir to %d,%s : %s",
         int(me), stl_string(coords).c_str(), z,
         int(addr2), stl_string(connect2).c_str(),
         dest2->to_string().c_str());



      connect_dim(params, z, objects[me], dest1, dest2);

    }
  }
}

void
hdtorus::configure_vc_routing(std::map<routing::algorithm_t, int>& m) const
{
  m[routing::minimal] = 2;
  m[routing::minimal_adaptive] = 2;
  m[routing::valiant] = 4;
  m[routing::ugal] = 6;
}

void
hdtorus::configure_geometric_paths(std::vector<int>& redundancies)
{
  int ndims = dimensions_.size();
  int ngeom_paths = ndims * 2 + endpoints_per_switch_; //2 for +/-
  redundancies.resize(ngeom_paths);
  for (int d=0; d < ndims; ++d){
    int pos_path = convert_to_port(d,pos);
    redundancies[pos_path] = red_[d];

    int neg_path = convert_to_port(d,neg);
    redundancies[neg_path] = red_[d];
  }

  configure_injection_geometry(redundancies);
}

int
hdtorus::convert_to_port(int dim, int dir) const
{
  return 2*dim + dir;
}

void
hdtorus::nearest_neighbor_partners(
  const coordinates &src_sw_coords,
  int port,
  std::vector<node_id>& partners) const
{
  int ndim = dimensions_.size();
  partners.resize(ndim * 2);
  coordinates tmp_coords(src_sw_coords);
  int idx = 0;
  for (int i=0; i < ndim; ++i) {
    int dim_size = dimensions_[i];

    //plus partner
    tmp_coords[i] = (src_sw_coords[i] + 1) % dim_size;
    partners[idx++] = node_addr(tmp_coords, port);

    //minus partner
    tmp_coords[i] = (src_sw_coords[i] + dim_size - 1) % dim_size;
    partners[idx++] = node_addr(tmp_coords, port);

    //reset coords
    tmp_coords[i] = src_sw_coords[i];
  }
}

void
hdtorus::tornado_recv_partners(
  const coordinates &src_sw_coords,
  int port,
  std::vector<node_id>& partners) const
{
  int ndim = dimensions_.size();
  partners.resize(1);
  coordinates tmp_coords(src_sw_coords);
  for (int i=0; i < ndim; ++i) {
    int dim_size = dimensions_[i];
    tmp_coords[i] = (src_sw_coords[i] - (dim_size - 1)/2 + dim_size) % dim_size;
  }
  partners[0] = node_addr(tmp_coords, port);
}

void
hdtorus::tornado_send_partners(
  const coordinates &src_sw_coords,
  int port,
  std::vector<node_id>& partners) const
{
  int ndim = dimensions_.size();
  partners.resize(1);
  coordinates tmp_coords(src_sw_coords);
  for (int i=0; i < ndim; ++i) {
    int dim_size = dimensions_[i];
    tmp_coords[i] = (src_sw_coords[i] + (dim_size - 1)/2) % dim_size;
  }
  partners[0] = node_addr(tmp_coords, port);
}

void
hdtorus::bit_complement_partners(
  const coordinates &src_sw_coords,
  int port,
  std::vector<node_id>& partners) const
{
  int ndim = dimensions_.size();
  partners.resize(1);
  coordinates tmp_coords(src_sw_coords);
  for (int i=0; i < ndim; ++i) {
    int dim_size = dimensions_[i];
    tmp_coords[i] = dim_size - src_sw_coords[i] - 1;
  }
  partners[0] = node_addr(tmp_coords, port);
}

}
} //end of namespace sstmac


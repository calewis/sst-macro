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
// fattree.cc: Implementation of fat tree networks.
//
// Author: Jeremiah Wilke <jjwilke@sandia.gov>
#include <sstream>
#include <sstmac/hardware/topology/fat_tree.h>
#include <sstmac/hardware/router/router.h>
#include <sstmac/backends/common/sim_partition.h>
#include <sprockit/sim_parameters.h>

#include <math.h>

namespace sstmac {
namespace hw {

SpktRegister("fattree | ftree", topology, fat_tree,
  "Fat tree topology with L levels and radix K.  This fat tree is actually implemented with commodity switches. Each level of the fat tree has the same number of switches.  This is equivalent to archetypal fat tree with fatter links being replaced by MORE links.");

SpktRegister("simple_fattree", topology, simple_fat_tree);

sprockit::sim_parameters*
abstract_fat_tree::override_params(sprockit::sim_parameters* params)
{
  std::vector<int> args;
  params->get_vector_param("geometry", args);
  if (args.size() != 2) {
    spkt_throw_printf(sprockit::input_error,
                     "fat_tree::override_params: geometry needs 2 parameters, got %d",
                     args.size());
  }
  l_ = args[0];
  k_ = args[1];
  if (!params->has_param("concentration")){
    params->add_param_override("concentration", k_);
  }

  sprockit::sim_parameters* rtr_params = params->get_optional_namespace("router");
  rtr_params->add_param_override("radix", k_);
  rtr_params->add_param_override("num_levels", l_);

  /**
     The first parameter gives the number of levels (L) in the fat tree.
     The second parameter gives the radix (K) of the fat tree.  The number of leaf switches
     is K^(L-1).  If L=3 and K=4, e.g., you have the following fat tree
     1 -> 4 -> 16 switches at the top.
     The number of nodes is K^L (if concentration parameter is not given).
     This gives 64 nodes for L=3, K=4.
   */
  toplevel_ = l_ - 1;
  numleafswitches_ = pow(k_, l_ - 1);

  return params;
}

abstract_fat_tree::abstract_fat_tree(sprockit::sim_parameters *params,
                                     InitMaxPortsIntra i1,
                                     InitGeomEjectID i2) :
  structured_topology(override_params(params), i1, i2)
{
}

void
fat_tree::minimal_route_to_switch(
  switch_id current_sw_addr,
  switch_id dest_sw_addr,
  structured_routable::path& path) const
{
  spkt_throw_printf(sprockit::unimplemented_error, "fattree::minimal_route_to_switch");
}

void
abstract_fat_tree::nodes_connected_to_injection_switch(switch_id swaddr,
                                                       std::vector<node_id>& nodes) const
{
  if (swaddr >= numleafswitches_){
    nodes.resize(0);
  } else {
    topology::nodes_connected_to_injection_switch(swaddr, nodes);
  }
}

void
abstract_fat_tree::nodes_connected_to_ejection_switch(switch_id swaddr,
                                                      std::vector<node_id>& nodes) const
{
  nodes_connected_to_injection_switch(swaddr, nodes);
}

fat_tree::fat_tree(sprockit::sim_parameters* params) :
  abstract_fat_tree(params,
                    InitMaxPortsIntra::I_Remembered,
                    InitGeomEjectID::I_Remembered)
{
  max_ports_intra_network_ = 2*k_;
  eject_geometric_id_ = max_ports_intra_network_;
}

void
fat_tree::productive_path(
  int dim,
  const coordinates &src,
  const coordinates &dst,
  structured_routable::path& path) const
{
  spkt_throw_printf(
    sprockit::illformed_error,
    "fattree::get_productive_dir should never be called."
    "productive outports are determined differently from other topologies");
}

int
fat_tree::upColumnConnection(int k, int myColumn, int upPort, int myBranchSize)
{
  upPort = upPort % k;
  int myReplicaID = myColumn % myBranchSize;
  int portStride = myBranchSize;
  int upBranchSize = myBranchSize*k;
  int myVirtualBranch = myColumn/myBranchSize;
  int upVirtualBranch = myVirtualBranch/k;
  int ret = upVirtualBranch*upBranchSize + upPort*portStride + myReplicaID;
  //printf("(c=%d,vc=%d,p=%d)->(c=%d,vc=%d)",
  //     myColumn, myVirtualColumn, upPort, ret, upVirtualColumn);
  return ret;
}

int
fat_tree::downColumnConnection(int k, int myColumn, int downPort, int myBranchSize)
{
  downPort = downPort % k;
  int myVirtualBranch = myColumn / myBranchSize;
  int myReplicaID = myColumn % myBranchSize;
  int lowerBranchSize = myBranchSize / k;
  int lowerReplicaID = myReplicaID % lowerBranchSize;
  return myVirtualBranch*k + downPort*lowerBranchSize + lowerReplicaID;
}

void
fat_tree::connect_objects(sprockit::sim_parameters* params, internal_connectable_map& objects)
{
  sprockit::sim_parameters* link_params = params->get_namespace("link");
  int branchSize = 1;
  int maxLevel = l_ - 2;
  for (int row=0; row <= maxLevel; ++row){
    int nColumns = numleafswitches_;
    for (int col=0; col < nColumns; ++col){
      int lower_id = switch_at_row_col(row, col);
      connectable* lower_switch = objects[switch_id(lower_id)];
      int myBranch = col / branchSize;
      for (int k=0; k < k_; ++k){
        int upColumn = upColumnConnection(k_, col, k, branchSize);
        int upper_id = switch_at_row_col(row+1,upColumn);

        int up_port = convert_to_port(up_dimension, k);
        int down_port = convert_to_port(down_dimension, myBranch % k_);

        top_debug("fattree: connecting up=(%d,%d:%d) to down=(%d,%d:%d)",
                row, col, up_port, row+1, upColumn, down_port);

        connectable* upper_switch = objects[switch_id(upper_id)];

        lower_switch->connect_output(
          link_params,
          up_port, //up is out and down is in... got it!??!
          down_port,
          upper_switch);
        upper_switch->connect_input(
          link_params,
          up_port,
          down_port,
          lower_switch);

        upper_switch->connect_output(
          link_params,
          down_port, //down is out and up is in... got it?!?
          up_port,
          lower_switch);
        lower_switch->connect_input(
          link_params,
          down_port,
          up_port,
          upper_switch);
      }
    }
    branchSize *= k_;
  }
}

void
fat_tree::configure_vc_routing(std::map<routing::algorithm_t, int> &m) const
{
  m[routing::minimal] = 2; //up and down
}

switch_id
fat_tree::switch_number(const coordinates &coords) const
{
  int row = coords[0];
  int col = coords[1];
  return row*numleafswitches_ + col;
}

void
fat_tree::compute_switch_coords(switch_id uid, coordinates& coords) const
{
  int row = uid / numleafswitches_;
  int col = uid % numleafswitches_;
  coords[0] = row;
  coords[1] = col;
}

void
fat_tree::minimal_route_to_coords(
  const coordinates &src_coords,
  const coordinates &dest_coords,
  structured_routable::path& path) const
{
  spkt_throw_printf(sprockit::unimplemented_error, "fattree::minimal_route_to_coords");
}

coordinates
fat_tree::neighbor_at_port(switch_id sid, int port)
{
  coordinates my_coords = switch_coords(sid);
  if (is_injection_port(port)){
    return my_coords;
  }

  int row = my_coords[0];
  int col = my_coords[1];
  int dir = port % k_;
  int dim = port / k_;

  int branchSize = 1;
  for (int l=0; l < row; ++l){
    branchSize *= k_;
  }

  if (dim == up_dimension){
    my_coords[0] = row+1;
    my_coords[1] = upColumnConnection(k_, col, port, branchSize);
  } else {
    my_coords[0] = row-1;
    my_coords[1] = downColumnConnection(k_, col, port, branchSize);
  }
  return my_coords;
}

int
fat_tree::convert_to_port(int dim, int dir) const
{
  return (dim * k_ + dir);
}

int
fat_tree::minimal_distance(const coordinates &src_coords,
                           const coordinates &dest_coords) const
{
  int srcRow = src_coords[0];
  int dstRow = dest_coords[0];
  int startRow = std::min(srcRow, dstRow);
  int branchSize = pow(k_, startRow);
  int srcCol = src_coords[1];
  int dstCol = dest_coords[1];
  int srcBranch = srcCol / branchSize;
  int dstBranch = dstCol / branchSize;
  int stopRow = startRow;
  //keep going up until these land in the same branch
  while (srcBranch != dstBranch){
    branchSize *= k_;
    srcBranch = srcCol / branchSize;
    dstBranch = dstCol / branchSize;
    ++stopRow;
  }

  int distance = (stopRow - srcRow)  + (stopRow - dstRow);
  return distance;
}

void
simple_fat_tree::partition(
  int* switches_per_lp,
  int *switch_to_lp,
  int *switch_to_thread,
  int& local_num_switches,
  int me,
  int nproc,
  int nthread,
  int noccupied)
{
  int nworkers = nproc * nthread;

  //partition all the occupied switches
  int sw_per_worker = noccupied / nworkers;
  if (noccupied % sw_per_worker) ++sw_per_worker;

  int switches_at_level = num_leaf_switches();
  int occ_at_level = noccupied;
  int swIdx = 0;
  int localIdx = 0;
  top_debug("simple fat tree k=%d l=%d partitioning %d switches onto %d procs x %d threads",
    k_, l_, num_switches(), nproc, nthread);
  for (int l=0; l < l_; ++l){
    top_debug("simple fat tree partitioning %d switches, %d occupied on level %d onto %d procs x %d threads",
      switches_at_level, occ_at_level, l, nproc, nthread);

    int switches_per_worker = occ_at_level / nworkers;
    if (occ_at_level % nworkers) ++switches_per_worker;
    for (int i=0; i < occ_at_level; ++i, ++swIdx){
      int worker = i / switches_per_worker;
      int lp = worker / nthread;
      switch_to_lp[swIdx] = lp;
      switches_per_lp[lp]++;
      if (lp == me){
        int thr = worker % nthread;
        switch_to_thread[localIdx] = thr;
        ++localIdx;
        top_debug("occupied switch %d(%d) assigned to proc %d, thread %d at local index %d", 
          swIdx, i, lp, thr, localIdx);
      }
    }

    int unocc_at_level = switches_at_level - occ_at_level;
    int switches_per_thread = unocc_at_level / nthread;
    if (unocc_at_level % nthread) ++switches_per_thread;
    for (int i=0; i < unocc_at_level; ++i, ++swIdx){
      //assign all these switches to the LAST proc
      int lp = nproc - 1;
      switch_to_lp[swIdx] = lp; //empty, assigned to zero
      switches_per_lp[lp]++;
      int thr = i / switches_per_thread;
      if (lp == me){
        switch_to_thread[localIdx] = thr;
        ++localIdx;
        top_debug("unoccupied switch %d(%d) assigned to proc %d, thread %d at local index %d", 
          swIdx, i, lp, thr, localIdx);
      }
    }

    switches_at_level /= k_;
    occ_at_level /= k_;
    occ_at_level = std::max(1, occ_at_level);
  }

  local_num_switches  = localIdx;
}

int
simple_fat_tree::num_hops(int srcLevel, int srcOffset, int dstLevel, int dstOffset) const
{
  int distance = 0;
  while (srcOffset != dstOffset){
    if (srcLevel <= dstLevel){
      srcOffset /= k_;
      ++srcLevel;
    } else {
      dstOffset /= k_;
      ++dstLevel;
    }
    ++distance;
  }

  return distance;
}

int
simple_fat_tree::num_hops_to_node(node_id src, node_id dst) const
{
  int src_sw = src / endpoints_per_switch_;
  int dst_sw = dst / endpoints_per_switch_;
  return num_hops(0, src_sw, 0, dst_sw);
}

int
simple_fat_tree::minimal_distance(
  const coordinates &src_coords,
  const coordinates &dest_coords) const
{
  abort();
  return num_hops(src_coords[0], src_coords[1], dest_coords[0], dest_coords[1]);
}

simple_fat_tree::simple_fat_tree(sprockit::sim_parameters *params) :
  abstract_fat_tree(params,
                    InitMaxPortsIntra::I_Remembered,
                    InitGeomEjectID::I_Remembered)
{
  int nswitches = numleafswitches_;
  int offset = 0;
  int level = 0;
  level_offsets_.resize(l_);
  num_switches_ = 0;
  while (nswitches >= 1){
    level_offsets_[level] = offset;
    top_debug("fat_tree: setting level offset %d to %d", level, offset);
    offset += nswitches;
    num_switches_ += nswitches;
    nswitches /= k_;
    level++;
  }
  top_debug("fat_tree: computed %d total switches on %d levels",
            num_switches_, l_);
  max_ports_intra_network_ = k_ + 1;
  eject_geometric_id_ = max_ports_intra_network_;

  if (params->has_param("tapering")){
    params->get_vector_param("tapering", tapering_);
  } else {
    tapering_.resize(toplevel_, 1.0);
  }

  if (tapering_.size() != toplevel_){
    spkt_throw_printf(sprockit::value_error,
      "fat_tree::tapering array of size %d is not of correct size %d",
      tapering_.size(), toplevel_);
  }
}

void
simple_fat_tree::build_internal_connectables(
  internal_connectable_map &connectables,
  connectable_factory factory,
  connectable_factory dummy_factory,
  sstmac::partition *part, int my_rank,
  sprockit::sim_parameters *params)
{
  int nswitches = numleafswitches_;
  int bw_multiplier = 1;
  sprockit::sim_parameters* xbar_params = params->get_namespace("xbar");
  double bw = xbar_params->get_bandwidth_param("bandwidth");
  for (int l=0; l < l_; ++l){
    int down_offset = level_offsets_[l];
    double tapering = tapering_[l];
    for (int s=0; s < nswitches; ++s){
      int down_id = down_offset + s;
      top_debug("Adding fat tree switch %d with multiplier %d", down_id, bw_multiplier);
      double xbar_bw = bw * bw_multiplier * tapering;
      (*xbar_params)["bandwidth"].setBandwidth(xbar_bw/1e9, "GB/s");
      if (part->lpid_for_switch(down_id) == my_rank){
        params->add_param_override("id", down_id);
        connectables[down_id] = factory(params, down_id);
      } else {
        connectables[down_id] = dummy_factory(params, down_id);
      }
    }
    nswitches /= k_;
    bw_multiplier *= k_;
  }
}

void
simple_fat_tree::connect_objects(sprockit::sim_parameters* params,
                                 internal_connectable_map &switches)
{
  sprockit::sim_parameters* link_params = params->get_namespace("link");
  sprockit::sim_parameters* xbar_params = params->get_namespace("xbar");
  int buffer_size = xbar_params->get_int_param("buffer_size");
  double bw = link_params->get_bandwidth_param("bandwidth");

  int nswitches = numleafswitches_;
  int bw_multiplier = 1;
  int stopLevel = l_ - 1;
  for (int l=0; l < stopLevel; ++l){
    int down_offset = level_offsets_[l];
    int up_offset = level_offsets_[l+1];
    double tapering = tapering_[l];
    for (int s=0; s < nswitches; ++s){
      int down_id = down_offset + s;
      int up_id = up_offset + s/k_;
      connectable* down_switch = switches[switch_id(down_id)];
      connectable* up_switch = switches[switch_id(up_id)];

      int down_switch_outport = k_;
      int down_switch_inport = down_switch_outport;
      int up_switch_outport = s % k_;
      int up_switch_inport = up_switch_outport;

      top_debug(
       "Connecting %d(%d):%d->%d(%d):%d between levels %d,%d with multiplier=%d, tapering=%12.8f",
       down_id, s, down_switch_outport,
       up_id, s/k_, up_switch_inport,
       l, l+1, bw_multiplier, tapering);


      sprockit::sim_parameters* port_params = setup_port_params(down_switch_outport,
                                                  buffer_size*bw_multiplier,
                                                  bw*bw_multiplier*tapering,
                                                  link_params,
                                                  params);

      down_switch->connect_output(
        port_params,
        down_switch_outport,
        up_switch_inport,
        up_switch);

      up_switch->connect_input(
         port_params,
        down_switch_outport,
        up_switch_inport,
        down_switch);

      top_debug(
       "Connecting %d(%d):%d->%d(%d):%d between levels %d,%d with multiplier=%d, tapering=%12.8f",
       up_id, s/k_, up_switch_outport,
       down_id, s, down_switch_inport,
       l, l+1, bw_multiplier, tapering);

      port_params = setup_port_params(up_switch_outport,
                                      buffer_size*bw_multiplier,
                                      bw*bw_multiplier*tapering,
                                      link_params,
                                      params);

      up_switch->connect_output(
        port_params,
        up_switch_outport,
        down_switch_inport,
        down_switch);

      down_switch->connect_input(
        port_params,
        up_switch_outport,
        down_switch_inport,
        up_switch);
    }
    nswitches /= k_;
    bw_multiplier *= k_;
  }

}

int
simple_fat_tree::level(switch_id sid) const
{
  int level_stop = level_offsets_.size() - 1;
  for (int i=0; i < level_stop; ++i){
    if (sid >= level_offsets_[i] && sid < level_offsets_[i+1]){
      return i;
    }
  }
  return toplevel_;
}

void
simple_fat_tree::minimal_route_to_switch(
  switch_id current_sw_addr,
  switch_id dest_sw_addr,
  structured_routable::path &path) const
{
  int src_level = level(current_sw_addr);
  int dst_level = level(dest_sw_addr);
  //question is whether I go up or down
  if (dst_level >= src_level){ //definitely have to go up
    path.outport = k_;
    path.vc = 0;
    top_debug("fat_tree: routing up to get to s=%d,l=%d from s=%d,l=%d",
            int(dest_sw_addr), dst_level,
            int(current_sw_addr), src_level);
  } else {
    //walk up from the destination switch - see if it hits the source
    int dstLevelOffset = dest_sw_addr - level_offsets_[dst_level];
    int dstLevelTmp = dst_level;
    int downPort;
    while (dstLevelTmp < src_level){
      downPort = dstLevelOffset % k_;
      dstLevelOffset /= k_;
      dstLevelTmp++;
    }
    int parentAtSrcLevel = dstLevelOffset + level_offsets_[src_level];
    if (parentAtSrcLevel == current_sw_addr){
      top_debug("fat_tree: routing down to get to s=%d,l=%d from s=%d,l=%d on port %d",
              int(dest_sw_addr), dst_level,
              int(current_sw_addr), src_level,
              downPort);
      //yep, we can hit the dest switch on the way down
      path.outport = downPort;
      path.vc = 1; //down, down
    } else {
      top_debug("fat_tree: routing up to get to s=%d,l=%d from s=%d,l=%d",
              int(dest_sw_addr), dst_level,
              int(current_sw_addr), src_level);
      path.outport = k_;
      path.vc = 0;
    }
  }
}

void
simple_fat_tree::compute_switch_coords(switch_id swid, coordinates &coords) const
{
  int srcLevel = level(swid);
  int srcOffset = swid - level_offsets_[srcLevel];
  coords.resize(2);
  coords[0] = srcLevel;
  coords[1] = srcOffset;
}

coordinates
simple_fat_tree::neighbor_at_port(switch_id sid, int port)
{
  coordinates coords(2);
  int srcLevel = level(sid);
  int srcOffset = sid - level_offsets_[srcLevel];
  if (port == k_){
    //going up
    coords[0] = srcLevel + 1;
    int dstOffset = srcOffset / k_;
    coords[1] = dstOffset;
  } else {
    //going down
    coords[0] = srcLevel - 1;
    int dstOffset = srcOffset * k_;
    coords[1] = dstOffset;
  }
  return coords;
}

int
simple_fat_tree::convert_to_port(int dim, int dir) const
{
  if (dim == up_dimension){
    return k_;
  } else {
    return dir;
  }
}

void
simple_fat_tree::productive_path(int dim,
  const coordinates &src,
  const coordinates &dst,
  structured_routable::path &path) const
{
  spkt_throw(sprockit::unimplemented_error,
     "simple_fat_tree should never route through productive_path function");
}

void
simple_fat_tree::configure_vc_routing(std::map<routing::algorithm_t, int> &m) const
{
  m[routing::minimal] = 2; //up and down
}

void
simple_fat_tree::minimal_route_to_coords(
  const coordinates &src_coords,
  const coordinates &dest_coords,
  structured_routable::path &path) const
{
  spkt_throw(sprockit::unimplemented_error,
     "simple_fat_tree should never route with coords");
}

switch_id
simple_fat_tree::switch_number(const coordinates &coords) const
{
  int level = coords[0];
  int offset = coords[1];
  return switch_id(level_offsets_[level] + offset);
}

}
} //end of namespace sstmac


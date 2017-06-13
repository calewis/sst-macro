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

#ifndef SSTMAC_COMMON_STATS_STATS_COMMON_H_INCLUDED
#define SSTMAC_COMMON_STATS_STATS_COMMON_H_INCLUDED

#include <sstmac/common/stats/stat_collector.h>
#include <sstmac/common/timestamp.h>
#include <iostream>
#include <fstream>
#include <map>
#include <sprockit/unordered.h>

namespace sstmac {


/**
 * this stat_collector class keeps a spy plot
 */
class stat_spyplot :
  public stat_collector
{
  FactoryRegister("ascii", stat_collector, stat_spyplot)
 public:
  virtual std::string
  to_string() const override {
    return "stat_spyplot";
  }

  virtual void simulation_finished(timestamp end) override;

  virtual void dump_to_file(const std::string& froot);

  virtual void dump_local_data() override;

  virtual void dump_global_data() override;

  virtual void reduce(stat_collector *coll) override;

  virtual void global_reduce(parallel_runtime *rt) override;

  virtual void clear() override;

  virtual ~stat_spyplot() {}

  virtual void add_one(int source, int dest);

  virtual void add(int source, int dest, long num);

  virtual stat_collector*
  do_clone(sprockit::sim_parameters* params) const override {
    return new stat_spyplot(params);
  }

  stat_spyplot(sprockit::sim_parameters* params) :
    max_dest_(0),
    stat_collector(params)
  {
  }

 protected:
  typedef spkt_unordered_map<int, long> long_map;
  typedef spkt_unordered_map<int, long_map> spyplot_map;
  spyplot_map vals_;
  int max_dest_;

};

/**
 * this stat_collector class keeps a spy plot, and outputs it as a png
 */
class stat_spyplot_png : public stat_spyplot
{
  FactoryRegister("png", stat_collector, stat_spyplot_png)
 public:
  stat_spyplot_png(sprockit::sim_parameters* params);

  std::string
  to_string() const override {
    return "stat_spyplot_png";
  }

  virtual void
  add(int source, int dest, long num) override;

  virtual void
  dump_to_file(const std::string& froot) override;

  void
  set_normalization(long max) {
    normalization_ = max;
  }

  virtual
  ~stat_spyplot_png() {
  }

  stat_collector*
  do_clone(sprockit::sim_parameters* params) const override {
    return new stat_spyplot_png(params);
  }

 private:
  long normalization_;
  bool fill_;

};

}

#endif

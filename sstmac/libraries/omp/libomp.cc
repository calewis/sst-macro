#include <sstmac/hardware/node/node.h>
#include <sstmac/hardware/processor/processor.h>
#include <sstmac/libraries/pthread/sstmac_pthread_impl.h>
#include <sstmac/replacements/omp.h>
#include <sstmac/software/api/api.h>
#include <sstmac/software/process/app.h>
#include <sstmac/software/process/operating_system.h>
#include <sstmac/software/process/thread.h>

#undef omp_init_lock
#undef omp_destroy_lock
#undef omp_set_lock
#undef omp_unset_lock
#undef omp_test_lock
#undef omp_get_num_threads
#undef omp_get_thread_num
#undef omp_get_max_threads
#undef omp_get_wtime
#undef omp_get_num_procs
#undef omp_set_num_threads
#undef omp_in_parallel
#undef omp_get_level
#undef omp_get_ancestor_thread_num

namespace sstmac {
namespace sw {

extern "C" void sstmac_omp_init_lock(sstmac_omp_lock_t *lock) {
  SSTMAC_pthread_mutex_init(lock, nullptr);
}

extern "C" void sstmac_omp_destroy_lock(sstmac_omp_lock_t *lock) {
  SSTMAC_pthread_mutex_destroy(lock);
}

extern "C" void sstmac_omp_set_lock(sstmac_omp_lock_t *lock) {
  SSTMAC_pthread_mutex_lock(lock);
}

extern "C" void sstmac_omp_unset_lock(sstmac_omp_lock_t *lock) {
  SSTMAC_pthread_mutex_unlock(lock);
}

extern "C" int sstmac_omp_test_lock(sstmac_omp_lock_t *lock) {
  return SSTMAC_pthread_mutex_trylock(lock);
}

extern "C" double sstmac_omp_get_wtime() {
  return sstmac::sw::OperatingSystem::currentOs()->now().sec();
}

extern "C" int sstmac_omp_get_thread_num() {
  sstmac::sw::Thread *t = sstmac::sw::OperatingSystem::currentThread();
  return t->ompGetThreadNum();
}

extern "C" int sstmac_omp_get_num_threads() {
  sstmac::sw::Thread *t = sstmac::sw::OperatingSystem::currentThread();
  return t->ompGetNumThreads();
}

extern "C" int sstmac_omp_get_max_threads() {
  sstmac::sw::Thread *t = sstmac::sw::OperatingSystem::currentThread();
  // for now, just return number of cores
  return t->ompGetMaxThreads();
}

extern "C" int sstmac_omp_get_num_procs() {
  sstmac::sw::OperatingSystem *os = sstmac::sw::OperatingSystem::currentOs();
  return os->node()->proc()->ncores();
}

extern "C" int sstmac_omp_get_proc_bind() {
  sstmac::sw::OperatingSystem *os = sstmac::sw::OperatingSystem::currentOs();

  using MapPair = decltype(*(os->env_begin()));
  auto match =
      std::find_if(os->env_begin(), os->env_end(),
                   [](MapPair &m) { return m.first == "OMP_PROC_BIND"; });

  if (match == os->env_end()) {
    return 0;
  }

  // enum is
  // {
  // false = 0
  // true = 1
  // master = 2
  // close = 3
  // spread = 4
  // }
  auto const &val = match->second;
  if (val == "false") {
    return 0;
  } else if (val == "true" || val == "spread") {
    return 4; // libomp returns omp_proc_spread if true is used.
  } else if (val == "master") {
    return 2;
  } else if (val == "close") {
    return 3;
  }

  return 0;
}

extern "C" int sstmac_omp_get_num_places() {
  sstmac::sw::OperatingSystem *os = sstmac::sw::OperatingSystem::currentOs();
  using MapPair = decltype(*(os->env_begin()));
  auto match =
      std::find_if(os->env_begin(), os->env_end(),
                   [](MapPair &m) { return m.first == "OMP_PLACES"; });

  sstmac::sw::Thread *t = sstmac::sw::OperatingSystem::currentThread();
  if (match == os->env_end()) {
    return t->ompGetMaxThreads();
  }

  auto const &val = match->second;
  if (val == "cores") {
    return os->node()->proc()->ncores();
  } else if (val == "threads") {
    return t->ompGetMaxThreads();
  } else if (val == "sockets") {
    return os->node()->nsocket();
  } else {
    spkt_throw_printf(sprockit::SpktError,
                      "Unrecognized value for OMP_PLACES.");
  }
}

extern "C" void sstmac_omp_set_num_threads(int nthr) {
  sstmac::sw::Thread *t = sstmac::sw::OperatingSystem::currentThread();
  t->ompSetNumThreads(nthr);
}

extern "C" int sstmac_omp_in_parallel() {
  sstmac::sw::Thread *t = sstmac::sw::OperatingSystem::currentThread();
  return t->ompInParallel();
}

extern "C" int sstmac_omp_get_level() {
  sstmac::sw::Thread *t = sstmac::sw::OperatingSystem::currentThread();
  return t->ompGetLevel();
}

extern "C" int sstmac_omp_get_ancestor_thread_num() {
  sstmac::sw::Thread *t = sstmac::sw::OperatingSystem::currentThread();
  return t->ompGetAncestorThreadNum();
}

} // namespace sw
} // namespace sstmac

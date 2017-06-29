#ifndef sstmac_null_buffer_h
#define sstmac_null_buffer_h

static void* sstmac_null_ptr = ((void*)0x123);

static inline bool isNonNullBuffer(void* buf){
  return buf && buf != sstmac_null_ptr;
}

static inline bool isNullBuffer(void* buf){
  return !(isNonNullBuffer(buf));
}

#endif

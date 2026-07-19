// ABOUTME: Host/device metadata rules for exact clean-prefix Qwen3.5 GDN length classes.
// ABOUTME: Centralizes hostile-length bounds so runtime dispatch and CPU tests share one contract.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <limits.h>

#ifdef __CUDACC__
#define Q35_META_HD __host__ __device__
#else
#define Q35_META_HD
#endif

Q35_META_HD static inline int q35_clean_gdn_length_class(int n) {
    if(n<=0 || n>64) return 0;
    if(n<=16) return 16;
    if(n<=32) return 32;
    if(n<=48) return 48;
    return 64;
}

static inline int q35_clean_gdn_metadata_valid(const int *offs,const int *lens,int m,int rows) {
    if(!offs || !lens || m<=0 || m>32 || rows<=0) return 0;
    int64_t end=0;
    for(int i=0;i<m;i++) {
        if(lens[i]<=0 || lens[i]>64 || offs[i]<0 || (int64_t)offs[i]!=end) return 0;
        end+=(int64_t)lens[i];
        if(end>INT_MAX || end>rows) return 0;
    }
    return end==rows;
}

#undef Q35_META_HD

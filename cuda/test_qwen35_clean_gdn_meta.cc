// ABOUTME: CPU unit tests for clean-GDN length classes, flattened offsets, and hostile bounds.
// ABOUTME: Covers zero/negative/overflow lengths and finite-vs-NaN omission preconditions.
#include <cassert>
#include <cmath>
#include <climits>
#include <limits>
#include "qwen35_clean_gdn_meta.h"

int main() {
    for(int n=1;n<=16;n++) assert(q35_clean_gdn_length_class(n)==16);
    for(int n=17;n<=32;n++) assert(q35_clean_gdn_length_class(n)==32);
    for(int n=33;n<=48;n++) assert(q35_clean_gdn_length_class(n)==48);
    for(int n=49;n<=64;n++) assert(q35_clean_gdn_length_class(n)==64);
    const int hostile[]={INT_MIN,-65,-1,0,65,66,INT_MAX};
    for(int n:hostile) assert(q35_clean_gdn_length_class(n)==0);

    int offs[]={0,1,17,49}, lens[]={1,16,32,15};
    assert(q35_clean_gdn_metadata_valid(offs,lens,4,64));
    int gap[]={0,2}; int two[]={1,1};
    assert(!q35_clean_gdn_metadata_valid(gap,two,2,3));
    int overlap[]={0,0};
    assert(!q35_clean_gdn_metadata_valid(overlap,two,2,2));
    int zero[]={0}; assert(!q35_clean_gdn_metadata_valid(zero,zero,1,0));
    int too_long[]={65}; assert(!q35_clean_gdn_metadata_valid(zero,too_long,1,65));
    assert(!q35_clean_gdn_metadata_valid(offs,lens,33,64));
    assert(!q35_clean_gdn_metadata_valid(nullptr,lens,4,64));

    // The CUDA clean kernel may omit 0*x only for finite x. NaN/Inf force the incumbent
    // WMMA path; signed zeros are finite and retain the explicit incumbent subtraction.
    assert(std::isfinite(0.0f) && std::isfinite(-0.0f));
    assert(!std::isfinite(std::numeric_limits<float>::infinity()));
    assert(!std::isfinite(std::numeric_limits<float>::quiet_NaN()));
    float p=0.0f,n=-0.0f;
    assert(!std::signbit(p-0.0f));
    assert(std::signbit(n-0.0f));
    return 0;
}

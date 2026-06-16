// CPU correctness oracle for NVFP4 reconstruction (nvfp4.h).
// build: g++ -std=c++17 -O2 -Wall -Wextra cuda/nvfp4_test.cc -o /tmp/nvfp4_test && /tmp/nvfp4_test
#include "nvfp4.h"
#include <cassert>
#include <cstdio>
#include <cmath>

static bool close(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

int main() {
    // ── E2M1 LUT: every code, both signs ──
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    for (uint32_t c = 0; c < 8; c++) {
        assert(close(nvfp4::e2m1_decode(c), mag[c]));
        assert(close(nvfp4::e2m1_decode(c | 8u), c == 0 ? 0.f : -mag[c]));  // sign bit
    }
    assert(close(nvfp4::e2m1_decode(7), 6.0f));    // max +
    assert(close(nvfp4::e2m1_decode(15), -6.0f));  // max -
    printf("e2m1 LUT: OK\n");

    // ── E4M3 decode at known bit patterns ──
    // S.EEEE.MMM, bias 7. 1.0 = 0.0111.000 = 0x38; 1.5 = 0.0111.100 = 0x3C; 2.0 = 0.1000.000 = 0x40
    assert(close(nvfp4::e4m3_decode(0x00), 0.0f));
    assert(close(nvfp4::e4m3_decode(0x38), 1.0f));
    assert(close(nvfp4::e4m3_decode(0x3C), 1.5f));
    assert(close(nvfp4::e4m3_decode(0x40), 2.0f));
    assert(close(nvfp4::e4m3_decode(0x30), 0.5f));   // 0.0110.000 = 2^-1
    assert(close(nvfp4::e4m3_decode(0xB8), -1.0f));  // sign set
    assert(close(nvfp4::e4m3_decode(0x7E), 448.0f, 0.5f)); // 0.1111.110 = max finite = 448
    assert(std::isnan(nvfp4::e4m3_decode(0x7F)));    // 0.1111.111 = NaN
    // smallest positive subnormal: 0.0000.001 = 1/8 * 2^-6 = 2^-9
    assert(close(nvfp4::e4m3_decode(0x01), std::ldexp(1.0f, -9), 1e-7f));
    printf("e4m3 decode: OK\n");

    // ── Full reconstruction: real = e2m1 * block * global ──
    // nibble=7 (+6), block=0x38 (1.0), global=0.5 → 3.0
    assert(close(nvfp4::reconstruct_host(7, 0x38, 0.5f), 3.0f));
    // nibble=15 (-6), block=0x40 (2.0), global=0.25 → -3.0
    assert(close(nvfp4::reconstruct_host(15, 0x40, 0.25f), -3.0f));
    // nibble=2 (+1), block=0x3C (1.5), global=2.0 → 3.0
    assert(close(nvfp4::reconstruct_host(2, 0x3C, 2.0f), 3.0f));
    printf("reconstruct: OK\n");

    // ── Packed nibble access: low=even, high=odd ──
    uint8_t row[2] = {0x72, 0x1F};  // byte0: low=2,high=7 ; byte1: low=15,high=1
    assert(nvfp4::nibble_at(row, 0) == 0x2);
    assert(nvfp4::nibble_at(row, 1) == 0x7);
    assert(nvfp4::nibble_at(row, 2) == 0xF);
    assert(nvfp4::nibble_at(row, 3) == 0x1);
    printf("nibble pack: OK\n");

    // ── End-to-end row dequant against an explicit reference (in=32 → 2 block scales) ──
    // Build a row where every element's nibble = 1 (+1.0), global = 1/(6*448) (a typical
    // weight_scale_2), block 0 scale = 0x38 (1.0), block 1 scale = 0x40 (2.0).
    uint8_t wrow[16]; for (int i = 0; i < 16; i++) wrow[i] = 0x22; // both nibbles = 2 (=+1.0)
    uint8_t srow[2] = {0x38, 0x40};
    float gs = 1.0f / (6.0f * 448.0f);
    float out[32];
    nvfp4::dequant_row_host(wrow, srow, gs, 32, out);
    for (int c = 0; c < 16; c++) assert(close(out[c], 1.0f * 1.0f * gs, 1e-9f));  // block 0
    for (int c = 16; c < 32; c++) assert(close(out[c], 1.0f * 2.0f * gs, 1e-9f)); // block 1
    printf("row dequant: OK\n");

    printf("ALL OK\n");
    return 0;
}

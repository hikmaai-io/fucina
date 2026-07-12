// ABOUTME: Gate for the DFlash draft-model device residency (upload + view binding) on real weights.
// ABOUTME: Uploads the real checkpoint and spot-checks device bytes against the safetensors file.
//
// Validates q35_dflash_residency_upload on the real z-lab/Qwen3.5-9B-DFlash: every per-layer and
// global weight view must point at device memory that byte-matches the source tensor, and the total
// slab size must equal the sum of tensor byte spans. SKIPs cleanly when the checkpoint is absent.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_residency.cu -o t && ./t
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "qwen35_dflash_residency.cuh"

static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }

// Compare a device BF16 view against the host source tensor bytes.
static bool view_matches(const __nv_bfloat16* dview, const st::Tensor* t){
    if(!dview||!t) return false;
    size_t n=t->nbytes; std::vector<uint8_t> back(n);
    if(cudaMemcpy(back.data(), dview, n, cudaMemcpyDeviceToHost)!=cudaSuccess){ cudaGetLastError(); return false; }
    return memcmp(back.data(), t->data, n)==0;
}

int main(int argc,char**argv){
    const char* def="/opt/spark/models/models--z-lab--Qwen3.5-9B-DFlash/"
                    "snapshots/5fc3b3d474760f18c516db87d84c37edbfd3ede6";
    std::string dir=(argc>1)?argv[1]:def;
    if(!exists(dir+"/model.safetensors")){ printf("SKIP — real DFlash checkpoint absent\n"); return 0; }

    std::ifstream f(dir+"/config.json"); std::stringstream ss; ss<<f.rdbuf(); std::string cj=ss.str();
    q35_dflash_residency R{}; std::string err;
    if(!qwen35dflash::parse_config(cj, R.geom, err)){ printf("FAIL parse: %s\n",err.c_str()); return 1; }
    st::Model M;
    if(!M.open((dir+"/model.safetensors").c_str(), err)){ printf("FAIL open: %s\n",err.c_str()); return 1; }
    if(!qwen35dflash::validate_tensors(M, R.geom, R.geom.V, err)){ printf("FAIL validate: %s\n",err.c_str()); return 1; }

    if(q35_dflash_residency_upload(&R, M, err)!=0){ printf("FAIL upload: %s\n",err.c_str()); return 1; }
    if(!R.ready){ printf("FAIL: not ready\n"); return 1; }

    // Expected slab bytes = sum of all uploaded tensor byte spans.
    size_t expect=0;
    auto span=[&](const std::string& n){ auto t=M.find(n); return t?t->nbytes:0; };
    expect += span("hidden_norm.weight") + span("norm.weight");
    if(R.geom.use_aux_hidden) expect += span("fc.weight");
    const char* per[]={"input_layernorm.weight","self_attn.q_proj.weight","self_attn.k_proj.weight",
        "self_attn.v_proj.weight","self_attn.o_proj.weight","self_attn.q_norm.weight",
        "self_attn.k_norm.weight","post_attention_layernorm.weight","mlp.gate_proj.weight",
        "mlp.up_proj.weight","mlp.down_proj.weight"};
    for(int l=0;l<R.geom.L;l++) for(const char* s:per) expect+=span("layers."+std::to_string(l)+"."+s);
    if(R.slab_bytes!=expect){ printf("FAIL: slab_bytes %zu != expected %zu\n",R.slab_bytes,expect); return 1; }

    int fails=0;
    if(!view_matches(R.hidden_norm, M.find("hidden_norm.weight"))){ printf("FAIL: hidden_norm view\n"); fails++; }
    if(!view_matches(R.final_norm, M.find("norm.weight"))){ printf("FAIL: final_norm view\n"); fails++; }
    if(R.geom.use_aux_hidden && !view_matches(R.fc, M.find("fc.weight"))){ printf("FAIL: fc view\n"); fails++; }
    for(int l=0;l<R.geom.L;l++){
        const q35_dflash_layer_w& w=R.layers[l];
        auto lk=[&](const char* s){ return "layers."+std::to_string(l)+"."+s; };
        struct V{ const __nv_bfloat16* d; const char* n; } vs[]={
            {w.input_norm,"input_layernorm.weight"},{w.q_proj,"self_attn.q_proj.weight"},
            {w.k_proj,"self_attn.k_proj.weight"},{w.v_proj,"self_attn.v_proj.weight"},
            {w.o_proj,"self_attn.o_proj.weight"},{w.q_norm,"self_attn.q_norm.weight"},
            {w.k_norm,"self_attn.k_norm.weight"},{w.post_norm,"post_attention_layernorm.weight"},
            {w.gate_proj,"mlp.gate_proj.weight"},{w.up_proj,"mlp.up_proj.weight"},{w.down_proj,"mlp.down_proj.weight"}};
        for(auto& v:vs) if(!view_matches(v.d, M.find(lk(v.n)))){ printf("FAIL: L%d %s view\n",l,v.n); fails++; }
    }

    double gb = R.slab_bytes/(1024.0*1024*1024);
    q35_dflash_residency_free(&R);
    if(fails){ printf("FAIL — DFlash residency (%d view mismatches)\n",fails); return 1; }
    printf("PASS — DFlash draft residency: %d layers uploaded to a %.3f GiB device slab; all global "
           "+ per-layer views byte-match the safetensors source\n", R.geom.L, gb);
    return 0;
}

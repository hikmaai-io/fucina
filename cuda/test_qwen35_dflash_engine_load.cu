// ABOUTME: Gate: the engine loads the resident DFlash draft model against the real target.
// ABOUTME: Creates the Qwen3.5-9B target engine, force-loads the draft substrate, asserts ready.
//
// Proves the in-engine lazy draft loader (gemma4_engine_q35_dflash_load) validates + uploads the
// real z-lab/Qwen3.5-9B-DFlash against the live target engine (geometry + vocab match) and marks
// the substrate resident. Requires the target GGUF (arg1) and FUCINA_QWEN35_DFLASH_PATH pointing at
// the draft snapshot. SKIPs cleanly if either is absent so weightless CI stays green.
//
// build via Makefile qwen35-dflash-engine-load-test.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include "gemma4_kernels.cuh"

static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }

int main(int argc,char**argv){
    const char* path=(argc>1)?argv[1]:"/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";
    const char* dpath=getenv("FUCINA_QWEN35_DFLASH_PATH");
    if(!exists(path)){ printf("SKIP — target checkpoint absent: %s\n",path); return 0; }
    if(!dpath||!exists(std::string(dpath)+"/model.safetensors")){ printf("SKIP — FUCINA_QWEN35_DFLASH_PATH unset or draft absent\n"); return 0; }

    gemma4_engine_t* eng=gemma4_engine_create(path,FORMAT_Q4_0,4096,0,0.90);
    if(!eng){ printf("FAIL: engine create\n"); return 2; }

    if(gemma4_engine_q35_dflash_load(eng)!=0){ printf("FAIL: draft load returned error\n"); gemma4_engine_destroy(eng); return 1; }
    if(!gemma4_engine_q35_dflash_ready(eng)){ printf("FAIL: draft not ready after load\n"); gemma4_engine_destroy(eng); return 1; }

    // Idempotent second load must succeed without re-uploading.
    if(gemma4_engine_q35_dflash_load(eng)!=0){ printf("FAIL: second load\n"); gemma4_engine_destroy(eng); return 1; }

    gemma4_engine_destroy(eng);
    printf("PASS — engine loaded the resident DFlash draft model against the real target (validated, "
           "geometry+vocab matched, ready)\n");
    return 0;
}

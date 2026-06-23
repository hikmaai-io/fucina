// test_e4b_forward.cu — validate the E4B forward pass against an HF reference.
//
// Reads /tmp/e4b_ref.bin (written by the HF dump: ids, scaled-embedding,
// post-layer-0 hidden, post-final-norm hidden, last-token softcapped logits),
// runs e4b_engine_forward_debug, and compares each checkpoint (cosine + rel L2)
// plus the argmax/top-5 of the logits. Exit 0 iff all gates pass.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include "e4b_engine.h"

static const char* kDir =
    "/opt/spark/models/hub/models--google--gemma-4-E4B-it/snapshots/"
    "fee6332c1abaafb77f6f9624236c63aa2f1d0187";

static void stats(const char* name, const float* a, const float* b, int n,
                  double* cos_out, double* rel_out){
    double dot=0,na=0,nb=0,diff=0;
    for(int i=0;i<n;i++){ double x=a[i],y=b[i]; dot+=x*y; na+=x*x; nb+=y*y; diff+=(x-y)*(x-y); }
    double cos=(na>0&&nb>0)?dot/(sqrt(na)*sqrt(nb)):1.0;
    double rel=(na>0)?sqrt(diff/na):0.0;
    printf("  %-22s cosine=%.6f  relL2=%.4f%%\n", name, cos, 100*rel);
    *cos_out=cos; *rel_out=rel;
}

int main(int argc, char** argv){
    const char* dir = (argc>1)?argv[1]:kDir;
    const char* ref = (argc>2)?argv[2]:"/tmp/e4b_ref.bin";

    FILE* f=fopen(ref,"rb");
    if(!f){ fprintf(stderr,"cannot open %s (run the HF dump first)\n",ref); return 1; }
    int32_t T,H,V;
    if(fread(&T,4,1,f)!=1||fread(&H,4,1,f)!=1||fread(&V,4,1,f)!=1){ fprintf(stderr,"bad ref\n"); return 1; }
    printf("reference: T=%d H=%d V=%d\n", T,H,V);
    std::vector<int32_t> ids(T);
    std::vector<float> emb(T*H), l0(T*H), fin(T*H), logits(V);
    if(fread(ids.data(),4,T,f)!=(size_t)T ||
       fread(emb.data(),4,(size_t)T*H,f)!=(size_t)T*H ||
       fread(l0.data(),4,(size_t)T*H,f)!=(size_t)T*H ||
       fread(fin.data(),4,(size_t)T*H,f)!=(size_t)T*H ||
       fread(logits.data(),4,V,f)!=(size_t)V){ fprintf(stderr,"short ref\n"); return 1; }
    fclose(f);

    e4b_engine_t* eng = e4b_engine_create(dir, 4096, 0);
    if(!eng){ fprintf(stderr,"FAIL: create\n"); return 1; }

    std::vector<float> e(T*H), c0(T*H), cf(T*H), cl(V);
    if(e4b_engine_forward_debug(eng, ids.data(), T, e.data(), c0.data(), cf.data(), cl.data())!=0){
        fprintf(stderr,"FAIL: forward\n"); e4b_engine_destroy(eng); return 1; }
    e4b_engine_destroy(eng);

    double cos,rel,worst_cos=1.0;
    stats("scaled-embedding", emb.data(), e.data(), T*H, &cos,&rel); worst_cos=fmin(worst_cos,cos);
    stats("post-layer-0",     l0.data(),  c0.data(),T*H, &cos,&rel); worst_cos=fmin(worst_cos,cos);
    stats("post-final-norm",  fin.data(), cf.data(),T*H, &cos,&rel); worst_cos=fmin(worst_cos,cos);
    stats("last-token logits",logits.data(), cl.data(), V, &cos,&rel); worst_cos=fmin(worst_cos,cos);

    // argmax agreement
    auto amax=[](const float* x,int n){ int b=0; for(int i=1;i<n;i++) if(x[i]>x[b]) b=i; return b; };
    int ra=amax(logits.data(),V), ma=amax(cl.data(),V);
    printf("  argmax: ref=%d mine=%d  ref_logit=%.3f mine_logit=%.3f\n",
           ra, ma, logits[ra], cl[ma]);

    const double GATE=0.99;
    bool pass = (worst_cos>=GATE) && (ra==ma);
    printf(pass ? "PASS: forward matches HF reference (worst cosine=%.6f, argmax agrees)\n"
                : "FAIL: worst cosine=%.6f or argmax mismatch\n", worst_cos);
    return pass?0:1;
}

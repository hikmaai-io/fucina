// ABOUTME: Losslessness gate for the target (1+K) verify-block forward vs sequential decode.
// ABOUTME: Per-row argmax of the block MUST equal token-by-token greedy decode from the same prefix.
//
// The DFlash greedy verify contract: scoring a (1+K)-token block in ONE forward and taking each
// row's argmax must yield exactly what sequential single-token greedy decode would produce at those
// positions. This gate warms a slot, records the sequential greedy continuation, then feeds the
// SAME tokens as a verify block and asserts the captured per-row argmax matches row-for-row, and
// that GDN state is rolled back (n_tokens unchanged). This is the core P4 losslessness proof for
// the target side. Uses the real target GGUF (arg1).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include "gemma4_kernels.cuh"

static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }
static int step1(gemma4_engine_t* eng,int slot,int32_t in){ int s[1]={slot}; int32_t i[1]={in},o[1]={-1}; if(gemma4_engine_step_batch(eng,s,i,1,o)!=0){ printf("step fail\n"); exit(2);} return o[0]; }

int main(int argc,char**argv){
    const char* path=(argc>1)?argv[1]:"/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";
    if(!exists(path)){ printf("SKIP — target checkpoint absent\n"); return 0; }
    gemma4_engine_t* eng=gemma4_engine_create(path,FORMAT_Q4_0,4096,0,0.90);
    if(!eng){ printf("FAIL create\n"); return 2; }

    int32_t prompt[5]={760,6511,314,9338,369}; const int NP=5, WARM=18, K=6, T=1+K;
    int32_t first=-1; int slot=gemma4_engine_seq_add(eng,prompt,NP,&first,0.f,0,1.f,0.f,1);
    if(slot<0){ printf("FAIL seq_add\n"); return 2; }
    int32_t cur=first; for(int i=0;i<WARM;i++) cur=step1(eng,slot,cur);
    // cur is the bonus/last-accepted token to be decoded next at the verify boundary.

    int base=gemma4_engine_seq_ntokens(eng,slot);
    // Sequential greedy reference: decode T tokens in place, recording the argmax at each position.
    // The block we will verify is [cur, d1..dK] where di are the sequential draft tokens; the
    // reference "next" argmax at row t is the token produced by decoding block[t].
    int32_t seqblock[16]; int32_t refnext[16];
    { int32_t c=cur; for(int t=0;t<T;t++){ seqblock[t]=c; c=step1(eng,slot,c); refnext[t]=c; } }
    // Roll the slot back to the boundary via a fresh state restore: re-add is simplest here — but we
    // must verify from the SAME base state. Use the GDN snapshot/commit(0) to rewind the WARM+T
    // decodes we just did back to `base`. We took no snapshot, so instead re-create the slot.
    gemma4_engine_seq_remove(eng,slot);

    // Rebuild the identical prefix (prompt + WARM decoded) so the slot is at `base` again.
    int32_t re=-1; int slot2=gemma4_engine_seq_add(eng,prompt,NP,&re,0.f,0,1.f,0.f,1);
    if(slot2<0){ printf("FAIL seq_add2\n"); return 2; }
    { int32_t c=re; for(int i=0;i<WARM;i++) c=step1(eng,slot2,c); }
    if(gemma4_engine_seq_ntokens(eng,slot2)!=base){ printf("FAIL: prefix rebuild base %d != %d\n",gemma4_engine_seq_ntokens(eng,slot2),base); return 1; }

    // Verify the SAME block; capture per-row argmax; assert rollback (n_tokens unchanged).
    int32_t am[16];
    if(gemma4_engine_q35_dflash_verify_block(eng,slot2,seqblock,T,am)!=0){ printf("FAIL verify_block\n"); return 1; }
    if(gemma4_engine_seq_ntokens(eng,slot2)!=base){ printf("FAIL: verify did not roll back (%d != %d)\n",gemma4_engine_seq_ntokens(eng,slot2),base); return 1; }

    int fails=0;
    for(int t=0;t<T;t++){ if(am[t]!=refnext[t]){ printf("FAIL row %d: verify argmax=%d seq=%d\n",t,am[t],refnext[t]); fails++; } }
    printf("verify block argmax:"); for(int t=0;t<T;t++) printf(" %d",am[t]); printf("\nseq next       :"); for(int t=0;t<T;t++) printf(" %d",refnext[t]); printf("\n");

    gemma4_engine_seq_remove(eng,slot2); gemma4_engine_destroy(eng);
    if(fails){ printf("FAIL — DFlash verify-block losslessness (%d/%d rows mismatch)\n",fails,T); return 1; }
    printf("PASS — DFlash target (1+K) verify block: per-row argmax == sequential greedy decode for "
           "all %d rows, GDN state rolled back (lossless verify contract)\n",T);
    return 0;
}

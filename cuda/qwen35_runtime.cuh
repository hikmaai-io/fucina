// Qwen3.5 continuous-batching runtime, state, prefill and decode orchestration.
// Internal implementation fragment: included once by gemma4_kernels.cu.
#pragma once

static int q35_fp4_setup(gemma4_engine_t *eng);   // fwd decl (defined after q35_proj_gemm)

static inline size_t q35_align_up(size_t n, size_t alignment = 256) {
    return (n + alignment - 1) & ~(alignment - 1);
}

static inline void bind_workspace(WorkspaceRef &ref,void *data,size_t bytes,WorkspaceKind kind) {
    ref.data=(uint8_t*)data; ref.bytes=bytes; ref.alignment=256; ref.kind=kind;
    ref.flags=0; ref.reserved=0;
}

static inline void q35_account_add(gemma4_engine_t *eng, size_t bytes) {
    eng->q35.committed_bytes += bytes;
    if (eng->q35.committed_bytes > eng->q35.reserved_bytes)
        eng->q35.reserved_bytes = eng->q35.committed_bytes;
    if (eng->q35.committed_bytes > eng->q35.peak_bytes)
        eng->q35.peak_bytes = eng->q35.committed_bytes;
}

// Check both the process policy and live physical headroom before growing a lazy allocation.
// A decline is non-fatal: attention callers use their low-memory scalar fallback.
static bool q35_can_grow(gemma4_engine_t *eng, size_t extra) {
    if (!extra) return true;
    size_t free_now=0, total_now=0; cudaMemGetInfo(&free_now, &total_now);
    const size_t safety = 256ull << 20;
    // On GB10, cudaMemGetInfo tracks raw MemFree and therefore treats reclaimable safetensors
    // page cache as unavailable. MemAvailable is the kernel's pressure-aware allocatable view;
    // use the larger value for the physical gate. cudaMalloc remains the final transactional gate.
    size_t physical=q35_physical_available(free_now);
    if (physical <= safety || extra > physical - safety) return false;
    // Exact owned bytes avoid billing file-cache/neighbor churn to this engine's util budget.
    size_t resident = eng->q35.model_bytes
        ? eng->q35.model_bytes + eng->q35.committed_bytes
        : (eng->free_mem > free_now ? eng->free_mem - free_now : eng->q35.committed_bytes);
    size_t budget = (size_t)(eng->gpu_mem_util * (double)(total_now ? total_now : eng->total_mem));
    return resident < budget && extra <= budget - resident;
}
static inline void q35_account_sub(gemma4_engine_t *eng, size_t bytes) {
    eng->q35.committed_bytes = bytes > eng->q35.committed_bytes ? 0 : eng->q35.committed_bytes - bytes;
}

// Roll back a failed first-time Qwen workspace transaction. ensure_ms_scratch is generic engine
// storage and is intentionally retained; every Qwen-owned allocation is returned to NULL so a
// later admission can retry after memory pressure subsides.
static void q35_workspace_rollback(gemma4_engine_t *eng) {
    for (int i = 0; i < 24; i++) { if (eng->q35.sb[i]) cudaFree(eng->q35.sb[i]); eng->q35.sb[i] = NULL; }
    #define Q35_ROLLBACK(p) do { if (eng->q35.p) cudaFree(eng->q35.p); eng->q35.p = NULL; } while (0)
    Q35_ROLLBACK(rowslot); Q35_ROLLBACK(chunk_scr); Q35_ROLLBACK(pf_pos); Q35_ROLLBACK(pf_tok);
    Q35_ROLLBACK(part_m); Q35_ROLLBACK(part_l); Q35_ROLLBACK(part_o);
    if (eng->q35.wbf16[0]) cudaFree(eng->q35.wbf16[0]); eng->q35.wbf16[0] = NULL;
    if (eng->q35.wbf16[1]) cudaFree(eng->q35.wbf16[1]); eng->q35.wbf16[1] = NULL;
    Q35_ROLLBACK(xbf16);
    for (int l = 0; l < GEMMA4_CAP_LAYERS; l++) {
        if (eng->q35.S[l]) cudaFree(eng->q35.S[l]); eng->q35.S[l] = NULL;
        if (eng->q35.ring[l]) cudaFree(eng->q35.ring[l]); eng->q35.ring[l] = NULL;
        if (eng->q35.Kc[l]) cudaFree(eng->q35.Kc[l]); eng->q35.Kc[l] = NULL;
        if (eng->q35.Vc[l]) cudaFree(eng->q35.Vc[l]); eng->q35.Vc[l] = NULL;
    }
    #undef Q35_ROLLBACK
    eng->q35.workspace_bytes = eng->q35.committed_bytes = eng->q35.reserved_bytes = 0;
}

// Lazily allocate the qwen35 M4 per-slot state arenas + per-row compute scratch (qwen35 only).
static int ensure_q35_scratch(gemma4_engine_t *eng) {
    if (eng->q35.ready) return 0;
    if (eng->cfg.arch != GEMMA4_ARCH_QWEN3_5) return -1;
    if (!eng->d_token_embd) {
        fprintf(stderr, "fucina: qwen35 M4: token_embd Q8_0 convert missing\n"); return -1;
    }
    // The qwen35 batched body reuses d_ms_pos (per-row position) + d_ms_outtok (per-row argmax),
    // both allocated by ensure_ms_scratch — required even though qwen35 uses no paged KV pool.
    if (ensure_ms_scratch(eng) != 0) {
        fprintf(stderr, "fucina: qwen35 M4: ensure_ms_scratch failed\n"); return -1;
    }
    const gemma4_model_config_t *c = &eng->cfg;
    int MS = eng->q35.capacity;
    const int L = c->n_layers;
    const int H=c->hidden_size, HD=M2_HEAD, NQ=c->n_heads, NKV=c->n_kv_global, INNER=c->ssm_inner_size, CONVD=(2*M2_KEYD+c->ssm_inner_size);
    const int KEYD=M2_KEYD, VALD=c->ssm_inner_size, NVH=(c->ssm_inner_size/M2_SD), SD=M2_SD, TSR=c->ssm_time_step_rank, CK=M2_CK;
    const int I = c->intermediate;
    // baked geometry must match the M2/M3 #defines (the mixer kernels bake head/GDN dims);
    // H and NKV are runtime (9B: 4096/4, 35B-A3B MoE: 2048/2).
    if (c->head_dim != HD || c->n_heads != NQ ||
        c->ssm_state_size != SD || c->ssm_group_count != M2_NKH || c->ssm_time_step_rank != NVH ||
        c->ssm_conv_kernel != CK || c->rotary_dim != M2_ROT) {
        fprintf(stderr, "fucina: qwen35 M4: geometry mismatch vs M2 constants\n"); return -1;
    }

    int dev = 0; cudaGetDevice(&dev);
    int optinMax = 49152;
    cudaDeviceGetAttribute(&optinMax, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    int capctx = optinMax / (int)sizeof(float) - 64;    // attn stores `maxctx` scores in shared
    int maxctx = (int)eng->context_size;
    if (const char *e = getenv("FUCINA_QWEN35_MAXCTX")) { int v = atoi(e); if (v > 0) maxctx = v; }
    if (maxctx < 1) maxctx = 4096;
    if (maxctx > capctx) {
        fprintf(stderr, "fucina: qwen35 M4 attn caps context at %d (store-in-shared); requested %d\n",
                capctx, maxctx);
        maxctx = capctx;
    }
    eng->q35.maxctx = maxctx;

    const size_t SBSZ[24] = {
        (size_t)H, (size_t)H, (size_t)2*NQ*HD, (size_t)NQ*HD, (size_t)NQ*HD,
        (size_t)NKV*HD, (size_t)NKV*HD, (size_t)NQ*HD, (size_t)H, (size_t)CONVD,
        (size_t)CONVD, (size_t)INNER, (size_t)TSR, (size_t)TSR, (size_t)TSR, (size_t)TSR,
        (size_t)KEYD, (size_t)KEYD, (size_t)VALD, (size_t)VALD, (size_t)INNER,
        (size_t)I, (size_t)I, (size_t)I
    };
    // Per-row compute scratch widens only to the largest tile this context can actually use.
    // Reserving all 8192 rows for a ctx-4096 engine doubled the dominant activation scratch.
    // Decode uses the first B<=MS rows; per-slot KV/state arenas below remain MS-sized.
    int PF = (maxctx < QWEN35_PF_TILE) ? maxctx : QWEN35_PF_TILE;
    if (PF < MS) PF = MS;
    size_t plan_free_before = 0, plan_total = 0;
    cudaMemGetInfo(&plan_free_before, &plan_total);
    const size_t plan_available_before=q35_physical_available(plan_free_before);
    int ok = 1;
    for (int i = 0; i < 24 && ok; i++)
        ok &= cudaMalloc(&eng->q35.sb[i], (size_t)PF * SBSZ[i] * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&eng->q35.rowslot, (size_t)PF * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&eng->q35.chunk_scr, (size_t)NVH * Q35_GDN_SCR * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&eng->q35.pf_pos, (size_t)PF * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&eng->q35.pf_tok, (size_t)PF * sizeof(int32_t)) == cudaSuccess;
    // S2a persistent per-slot decode state (last token id + next position), capacity-indexed.
    ok = ok && cudaMalloc(&eng->q35.d_slot_tok, (size_t)MS * sizeof(int32_t)) == cudaSuccess;
    ok = ok && cudaMalloc(&eng->q35.d_slot_pos, (size_t)MS * sizeof(int)) == cudaSuccess;
    // Flash-decoding attention split count S: capture-stable, derived from q35_maxctx only (target
    // ~512 positions/split so each block's serial loop is short and NQ*S fills the 48 SMs at B=1).
    {
        const int TILE_POS = 512, S_MAX = 64;
        int S = (maxctx + TILE_POS - 1) / TILE_POS;
        if (S < 1) S = 1; if (S > S_MAX) S = S_MAX;
        eng->q35.attn_splits = S;
        eng->q35.attn_tile = (maxctx + S - 1) / S;   // positions per split → shared-score floats
        // Partials sized by the RUNTIME head count (27B-dense: NQ=24 > M2_NQ=16 — sizing with the
        // baked constant under-allocated 1.5×, an illegal access once B·NQ·S crossed the arena).
        const size_t np = (size_t)MS * NQ * S;
        ok = ok && cudaMalloc(&eng->q35.part_m, np * sizeof(float)) == cudaSuccess;
        ok = ok && cudaMalloc(&eng->q35.part_l, np * sizeof(float)) == cudaSuccess;
        ok = ok && cudaMalloc(&eng->q35.part_o, np * M2_HEAD * sizeof(float)) == cudaSuccess;
    }
    // Tensor-core prefill GEMM scratch: dequant each projection weight → BF16 (ping-pong for
    // dequant/compute overlap), and the per-tile activation → BF16. Sized by the largest qwen35
    // projection and the widest GEMM in_dim ACROSS ALL variants (9B dense: FFN I×H dominates;
    // 35B MoE: I is the tiny shared-expert inter, so in_qkv CONVD×H is the largest).
    size_t wbf_max = (size_t)I * H;
    if ((size_t)CONVD * H > wbf_max) wbf_max = (size_t)CONVD * H;
    if ((size_t)2*NQ*HD * H > wbf_max) wbf_max = (size_t)2*NQ*HD * H;
    int xin_max = I;
    if (INNER > xin_max) xin_max = INNER;
    if (NQ*HD > xin_max) xin_max = NQ*HD;
    if (H > xin_max) xin_max = H;
    const size_t xbf_max = (size_t)PF * xin_max;

    // Build the Phase-1 memory plan BEFORE committing Qwen allocations. Capacity is reduced to
    // what fits both physical free memory and --gpu-mem-util, leaving graph/MoE working headroom.
    size_t workspace = (size_t)PF * (sizeof(int) * 2 + sizeof(int32_t));
    for (int i = 0; i < 24; i++) workspace += (size_t)PF * SBSZ[i] * sizeof(float);
    workspace += (size_t)NVH * Q35_GDN_SCR * sizeof(float);
    workspace += 2 * wbf_max * sizeof(__nv_bfloat16) + xbf_max * sizeof(__nv_bfloat16);
    workspace += 2ull * (size_t)L * MS * sizeof(void*); // two state pointer tables per layer
    workspace += (size_t)MS * NQ * eng->q35.attn_splits *
                 (2 * sizeof(float) + M2_HEAD * sizeof(float));
    size_t per_recur = 0, kv_per_token = 0;
    const size_t recurrent_s_bytes = (size_t)NVH * SD * SD * sizeof(__nv_bfloat16);
    const size_t recurrent_ring_bytes = (size_t)CONVD * (CK - 1) * sizeof(float);
    for (int l = 0; l < L; l++) {
        if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
            kv_per_token += 2ull * (size_t)NKV * HD * sizeof(__half);
        } else {
            // One fixed-state slab per slot, with independently aligned layer views. Besides
            // reducing 48 cudaMallocs/slot to one, this makes admission genuinely transactional:
            // either every GDN layer has state or none does.
            per_recur = q35_align_up(per_recur);
            per_recur += recurrent_s_bytes;
            per_recur = q35_align_up(per_recur);
            per_recur += recurrent_ring_bytes;
        }
    }
    per_recur = q35_align_up(per_recur);
    const size_t per_kv = kv_per_token * (size_t)maxctx; // maximum growth/accounting bytes
    // Admission reserves a production-typical context per slot, not maxctx for every slot. FULL
    // KV grows transactionally; a sequence beyond this reservation may continue while budget is
    // available, otherwise q35_can_grow declines before allocation and serving stops gracefully.
    // Set FUCINA_QWEN35_SLOT_CTX=maxctx to restore the old worst-case reservation policy.
    int slotctx=maxctx < 8192 ? maxctx : 8192;
    if (const char *e=getenv("FUCINA_QWEN35_SLOT_CTX")) { int v=atoi(e); if(v>0) slotctx=v; }
    if (slotctx>maxctx) slotctx=maxctx; if(slotctx<256) slotctx=256;
    eng->q35.reserved_context=slotctx;
    const size_t reserved_kv=kv_per_token*(size_t)slotctx;
    const size_t per_slot = per_recur + reserved_kv;
    const size_t reserve = 512ull << 20;
    size_t physical_room = plan_available_before > reserve ? plan_available_before - reserve : 0;
    // The exact Qwen model ledger is stable across cold/warm mmap cache state. workspace is
    // subtracted explicitly below, so resident here is model-only.
    size_t resident = eng->q35.model_bytes
        ? eng->q35.model_bytes
        : (eng->free_mem > plan_free_before ? eng->free_mem - plan_free_before : 0);
    size_t util_budget = (size_t)(eng->gpu_mem_util * (double)(plan_total ? plan_total : eng->total_mem));
    size_t budget_room = util_budget > resident ? util_budget - resident : 0;
    size_t room = physical_room < budget_room ? physical_room : budget_room;
    int fit = (room > workspace && per_slot) ? (int)((room - workspace) / per_slot) : 0;
    fprintf(stderr,
            "fucina: qwen35 memory-plan inputs: requested=%d slotctx=%d maxctx=%d "
            "cuda-free=%.2f GiB available=%.2f GiB model-resident=%.2f GiB (%s) "
            "workspace=%.2f GiB reserve/slot=%.2f GiB (max %.2f) "
            "physical-room=%.2f GiB budget-room=%.2f GiB chosen-room=%.2f GiB fit=%d\n",
            MS,slotctx,maxctx,plan_free_before/(1024.0*1024*1024),
            plan_available_before/(1024.0*1024*1024),resident/(1024.0*1024*1024),
            eng->q35.model_bytes ? "ledger" : "free-delta",
            workspace/(1024.0*1024*1024),per_slot/(1024.0*1024*1024),
            (per_recur+per_kv)/(1024.0*1024*1024),physical_room/(1024.0*1024*1024),
            budget_room/(1024.0*1024*1024),room/(1024.0*1024*1024),fit);
    if (fit < MS) {
        if (fit < 1) {
            fprintf(stderr, "fucina: qwen35 memory plan cannot fit one slot: workspace %.2f GiB, "
                    "state/slot %.2f GiB, room %.2f GiB\n",
                    workspace/(1024.0*1024*1024), per_slot/(1024.0*1024*1024), room/(1024.0*1024*1024));
            q35_workspace_rollback(eng); return -1;
        }
        fprintf(stderr, "fucina: qwen35 capacity reduced %d -> %d by guaranteed memory plan\n", MS, fit);
        MS = fit; eng->q35.capacity = fit;
    }
    eng->q35.workspace_bytes = workspace;
    eng->q35.per_slot_recurrent_bytes = per_recur;
    eng->q35.per_slot_kv_bytes = per_kv;
    eng->q35.reserved_slot_kv_bytes = reserved_kv;
    eng->q35.reserved_bytes = workspace + (size_t)MS * per_slot + eng->q35.jspace_bytes;

    ok = ok && cudaMalloc(&eng->q35.wbf16[0], wbf_max * sizeof(__nv_bfloat16)) == cudaSuccess;
    ok = ok && cudaMalloc(&eng->q35.wbf16[1], wbf_max * sizeof(__nv_bfloat16)) == cudaSuccess;
    ok = ok && cudaMalloc(&eng->q35.xbf16,    xbf_max * sizeof(__nv_bfloat16)) == cudaSuccess;
    // Allocate only stable device pointer tables here. Per-slot recurrent/KV storage is created
    // transactionally by q35_slot_state_ensure on first admission and retained for slot reuse.
    for (int l = 0; l < L && ok; l++) {
        if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
            ok &= cudaMalloc(&eng->q35.Kc[l], (size_t)MS*sizeof(__half*)) == cudaSuccess;
            ok &= cudaMalloc(&eng->q35.Vc[l], (size_t)MS*sizeof(__half*)) == cudaSuccess;
            if (ok) {
                ok &= cudaMemset(eng->q35.Kc[l], 0, (size_t)MS*sizeof(__half*)) == cudaSuccess;
                ok &= cudaMemset(eng->q35.Vc[l], 0, (size_t)MS*sizeof(__half*)) == cudaSuccess;
            }
        } else {
            ok &= cudaMalloc(&eng->q35.S[l], (size_t)MS*sizeof(__nv_bfloat16*)) == cudaSuccess;
            ok &= cudaMalloc(&eng->q35.ring[l], (size_t)MS*sizeof(float*)) == cudaSuccess;
            if (ok) {
                ok &= cudaMemset(eng->q35.S[l], 0, (size_t)MS*sizeof(__nv_bfloat16*)) == cudaSuccess;
                ok &= cudaMemset(eng->q35.ring[l], 0, (size_t)MS*sizeof(float*)) == cudaSuccess;
            }
        }
    }
    size_t smGDN = ((size_t)SD * SD + 3 * SD) * sizeof(float);
    size_t smGDNchunk = ((size_t)SD * SD + M2_CHUNK * SD + 3 * M2_CHUNK) * sizeof(float);
    ok = ok && cudaFuncSetAttribute(qwen35_b_gdn_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smGDN) == cudaSuccess;
    ok = ok && cudaFuncSetAttribute(qwen35_b_gdn_chunk_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smGDNchunk) == cudaSuccess;
    ok = ok && cudaFuncSetAttribute(qwen35_b_attn_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, maxctx * (int)sizeof(float)) == cudaSuccess;
    if (!ok) {
        fprintf(stderr, "fucina: qwen35 M4 scratch alloc/opt-in failed — transaction rolled back\n");
        q35_workspace_rollback(eng); cudaGetLastError(); return -1;
    }
    bind_workspace(eng->q35.routing_workspace,eng->q35.rowslot,(size_t)PF*sizeof(int),WorkspaceKind::DECODE);
    bind_workspace(eng->q35.gdn_workspace,eng->q35.chunk_scr,
                   (size_t)NVH*Q35_GDN_SCR*sizeof(float),WorkspaceKind::RECURRENT_STATE);
    bind_workspace(eng->q35.prefill_position_workspace,eng->q35.pf_pos,
                   (size_t)PF*sizeof(int),WorkspaceKind::PREFILL);
    bind_workspace(eng->q35.prefill_token_workspace,eng->q35.pf_tok,
                   (size_t)PF*sizeof(int32_t),WorkspaceKind::PREFILL);
    const size_t attention_partials=(size_t)MS*NQ*eng->q35.attn_splits;
    bind_workspace(eng->q35.attention_workspace[0],eng->q35.part_m,
                   attention_partials*sizeof(float),WorkspaceKind::ATTENTION);
    bind_workspace(eng->q35.attention_workspace[1],eng->q35.part_l,
                   attention_partials*sizeof(float),WorkspaceKind::ATTENTION);
    bind_workspace(eng->q35.attention_workspace[2],eng->q35.part_o,
                   attention_partials*M2_HEAD*sizeof(float),WorkspaceKind::ATTENTION);
    for(int i=0;i<24;i++)
        bind_workspace(eng->q35.decode_workspace[i],eng->q35.sb[i],
                       (size_t)PF*SBSZ[i]*sizeof(float),WorkspaceKind::DECODE);
    bind_workspace(eng->q35.prefill_weight_workspace[0],eng->q35.wbf16[0],
                   wbf_max*sizeof(__nv_bfloat16),WorkspaceKind::PREFILL);
    bind_workspace(eng->q35.prefill_weight_workspace[1],eng->q35.wbf16[1],
                   wbf_max*sizeof(__nv_bfloat16),WorkspaceKind::PREFILL);
    bind_workspace(eng->q35.prefill_input_workspace,eng->q35.xbf16,
                   xbf_max*sizeof(__nv_bfloat16),WorkspaceKind::PREFILL);
    eng->q35.committed_bytes = eng->q35.workspace_bytes + eng->q35.jspace_bytes;
    eng->q35.peak_bytes = eng->q35.committed_bytes;
    if (getenv("FUCINA_NO_BATCHED_GRAPH") || eng->fp4m_ssd_stream) eng->q35.graph_enabled = 0;
    // Resident BF16 prefill-weight cache: kills the per-prefill ~45% dequant cost at the price of
    // ~2x(weight bytes) BF16 (≈18 GB for 9B). Default-on ONLY when device memory comfortably
    // allows (memory-constrained clients keep the per-tile dequant). Env force: 1=on, 0=off.
    {
        size_t freeb = 0, totalb = 0; cudaMemGetInfo(&freeb, &totalb);
        // BF16 cache size from the ACTUAL dims (the GEMM'd per-layer projections): GDN in_qkv/
        // in_z/out + FULL q/k/v/o + dense FFN gate/up/down. 9B: ~0.5 GB/layer; 35B MoE: ~70 MB/
        // layer (experts are never wcached — they ride the grouped FP8 GEMM). Enable when the
        // cache plus a 5 GB working margin fits in free device memory.
        const size_t per_layer = 2ull * (size_t)H *
            ((size_t)CONVD + 2ull*INNER + 3ull*(size_t)I + 4ull*(size_t)NQ*HD);
        const size_t need = (size_t)L * per_layer;
        // 3 GiB working margin covers the MoE tc-prefill scratch (dequant slabs + gathered
        // activations, ~2 GiB) on top of the fixed KV/arenas already reflected in `freeb`.
        // The cache is a one-time alloc and each lazy per-layer build self-heals to off on
        // failure, so a tight-but-passing decision never destabilizes serving. (Was 5 GiB —
        // too conservative: it kept the ~350 ms/prefill mixer dequant on even with room.)
        const size_t available=q35_physical_available(freeb);
        const bool physical_ok = available > need + (size_t)3ull * 1024 * 1024 * 1024;
        const bool budget_ok = eng->q35.reserved_bytes + need <= room;
        eng->q35.wcache_on = (physical_ok && budget_ok) ? 1 : 0;
        const char *wcache_env = getenv("FUCINA_QWEN35_WCACHE");
        if (wcache_env) eng->q35.wcache_on = (atoi(wcache_env) != 0);
        if (eng->q35.wcache_on) eng->q35.reserved_bytes += need;
        fprintf(stderr, "fucina: qwen35 prefill weight-cache %s (decision=%s, need %.2f GiB, "
                "cuda-free %.2f GiB, available %.2f GiB, physical=%s, budget=%s, "
                "reserved-before %.2f GiB, room %.2f GiB)\n",
                eng->q35.wcache_on ? "ON" : "off", wcache_env ? "env" : "auto",
                need / (1024.0*1024*1024), freeb / (1024.0*1024*1024),
                available / (1024.0*1024*1024),
                physical_ok ? "fit" : "no-fit", budget_ok ? "fit" : "no-fit",
                (eng->q35.reserved_bytes - (eng->q35.wcache_on ? need : 0))/(1024.0*1024*1024),
                room/(1024.0*1024*1024));
    }
    q35_fp4_setup(eng);   // NVFP4 4-bit tensor-core prefill GEMM (preferred; BF16 cache is fallback)
    eng->q35.ready = 1;
    fprintf(stderr,
            "fucina: qwen35 memory plan ready: maxctx=%d slotctx=%d slots=%d/%d workspace=%.2f GiB "
            "reserve/slot=%.2f GiB (recurrent %.2f + KV %.2f; max-KV %.2f) committed=%.2f GiB reserved=%.2f GiB\n",
            maxctx, slotctx, MS, GEMMA4_MAX_SEQS,
            eng->q35.workspace_bytes/(1024.0*1024*1024), per_slot/(1024.0*1024*1024),
            per_recur/(1024.0*1024*1024), reserved_kv/(1024.0*1024*1024),
            per_kv/(1024.0*1024*1024),
            eng->q35.committed_bytes/(1024.0*1024*1024),
            eng->q35.reserved_bytes/(1024.0*1024*1024));
    return 0;
}

// Kernel-launch-only B-row hybrid forward (all per-step inputs DEVICE-resident → graph-safe).
// Advances each row's per-slot GDN state / conv ring / FULL-layer K/V; leaves B logit rows in
// d_sb[11]; when want_argmax, appends the per-row greedy argmax into d_ms_outtok.
static void qwen35_decode_multiseq_body(gemma4_engine_t *eng, int B, int want_argmax,
                                        int splice, cudaStream_t st) {
    const gemma4_model_config_t *c = &eng->cfg;
    const int H=c->hidden_size, HD=M2_HEAD, NQ=c->n_heads, NKV=c->n_kv_global;
    const int INNER=c->ssm_inner_size, CONVD=(2*M2_KEYD+c->ssm_inner_size), NKH=M2_NKH, NVH=(c->ssm_inner_size/M2_SD), SD=M2_SD, TSR=c->ssm_time_step_rank, ROT=M2_ROT;
    const int I = c->intermediate, VOC = c->vocab_size, L = c->n_layers;
    const int maxctx = eng->q35.maxctx;
    const float eps = 1e-6f;
    const size_t smGDN = ((size_t)SD * SD + 3 * SD) * sizeof(float);
    // (decode attention is flash-decoding now — partial+combine, no maxctx-score shared store)
    auto Wq = [&](uint64_t off) -> const uint8_t* { return weight_fp8(eng, off); };
    auto Wf = [&](uint64_t off) -> const float*   {
        return (const float*)(eng->d_weights + (off - eng->tdata_start)); };
    int   *d_pos  = eng->d_ms_pos;
    int   *d_slot = workspace_data<int>(eng->q35.routing_workspace);
    int32_t *d_tok = (int32_t*)eng->d_sb[0];
    auto ws=[&](int id){ return workspace_data<float>(eng->q35.decode_workspace[id]); };
    float *x=ws(Q35_X), *xn=ws(Q35_XN), *qg=ws(Q35_QG),
          *qb=ws(Q35_QB), *gate=ws(Q35_GATE), *kb=ws(Q35_KB),
          *vb=ws(Q35_VB), *attn=ws(Q35_ATTN), *mix=ws(Q35_MIX),
          *qkv=ws(Q35_QKV), *conv=ws(Q35_CONV), *zc=ws(Q35_ZC),
          *ac=ws(Q35_AC), *bc=ws(Q35_BC), *gg=ws(Q35_GG),
          *bb=ws(Q35_BB), *qh=ws(Q35_QH), *kh=ws(Q35_KH),
          *vh=ws(Q35_VH), *core=ws(Q35_CORE), *gnorm=ws(Q35_GNORM),
          *fg=ws(Q35_FG), *fu=ws(Q35_FU), *fa=ws(Q35_FA);
    auto grid1d = [](size_t n){ return (unsigned)((n + 255) / 256); };

    // S2a — GPU-native input splice: derive this step's input_ids/positions from persistent
    // per-slot state through the row→slot map, ON-GPU, so the step (below) replays with no host
    // knowledge of the previously sampled token. The host still uploads d_slot (the scheduler's
    // batch membership); it no longer needs to supply the sampled value. Pure gather ⇒ the spliced
    // (tok,pos) is bit-identical to what the host copy path would have written.
    if (splice)
        qwen35_splice_inputs_kernel<<<grid1d((size_t)B),256,0,st>>>(
            d_tok, d_pos, eng->q35.d_slot_tok, eng->q35.d_slot_pos, d_slot, B);

    embed_w(eng, x, eng->d_token_embd, d_tok, B, H, st);

    for (int l = 0; l < L; l++) {
        const auto &T = eng->tensors.layers[l];
        rms_norm_rows_kernel<<<B,256,32*sizeof(float),st>>>(xn, x, Wf(T.attn_norm), H, B, eps);
        moe_profile_activation(eng, l, 0, xn, (size_t)B*H, st);
        if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
            gemv_batched_w(eng, qg, T.ref_q, xn, B, st);
            gemv_batched_w(eng, kb, T.ref_k, xn, B, st);
            gemv_batched_w(eng, vb, T.ref_v, xn, B, st);
            m2_split_query_gate_kernel<<<grid1d((size_t)B*NQ*HD),256,0,st>>>(qb, gate, qg, B, NQ);
            per_head_rms_norm_rows_kernel<<<dim3(NQ,B),256,32*sizeof(float),st>>>(qb, Wf(T.attn_q_norm), NQ, HD, B, eps);
            per_head_rms_norm_rows_kernel<<<dim3(NKV,B),256,32*sizeof(float),st>>>(kb, Wf(T.attn_k_norm), NKV, HD, B, eps);
            qwen35_b_rope_kernel<<<dim3((ROT/2+31)/32,NQ,B),32,0,st>>>(qb, NQ, d_pos, B);
            qwen35_b_rope_kernel<<<dim3((ROT/2+31)/32,NKV,B),32,0,st>>>(kb, NKV, d_pos, B);
            qwen35_b_kv_write_kernel<<<dim3(grid1d((size_t)NKV*HD),B),256,0,st>>>(
                eng->q35.Kc[l], eng->q35.Vc[l], kb, vb, d_pos, d_slot, maxctx, B, NKV);
            // Flash-decoding: NQ*S partial blocks (fill the 48 SMs at B=1) → combine. The old
            // single-block qwen35_b_attn_kernel remains only on the prefill-continuation path (T rows).
            {
                int S = eng->q35.attn_splits;
                size_t smP = (size_t)eng->q35.attn_tile * sizeof(float);
                qwen35_flash_partial_kernel<<<dim3(NQ,B,S),256,smP,st>>>(
                    qb, eng->q35.Kc[l], eng->q35.Vc[l], d_pos, d_slot, maxctx, B, NKV, S,
                    workspace_data<float>(eng->q35.attention_workspace[0]),
                    workspace_data<float>(eng->q35.attention_workspace[1]),
                    workspace_data<float>(eng->q35.attention_workspace[2]), NQ);
                qwen35_flash_combine_kernel<<<dim3(NQ,B),256,0,st>>>(
                    attn, workspace_data<float>(eng->q35.attention_workspace[0]),
                    workspace_data<float>(eng->q35.attention_workspace[1]),
                    workspace_data<float>(eng->q35.attention_workspace[2]), B, S, NQ);
            }
            m2_sigmoid_gate_mul_kernel<<<grid1d((size_t)B*NQ*HD),256,0,st>>>(attn, gate, B*NQ*HD);
            moe_profile_activation(eng, l, 1, attn, (size_t)B*NQ*HD, st);
            gemv_batched_w(eng, mix, T.ref_o, attn, B, st);
        } else {
            // in_qkv requantized Q5_K→Q8_0 at load → native dp4a GEMV (P5), no per-step fp32 dequant.
            gemv_batched_w(eng, qkv, T.ssm.ref_in_qkv, xn, B, st);
            gemv_batched_w(eng, zc, T.ssm.ref_in_z, xn, B, st);
            if (eng->format == FORMAT_FP8_BLOCK) {   // in_a/in_b are f32 (oracle parity)
                m2_gemm(ac, xn, Wf(T.ssm.in_a), B, H, TSR, st);   // alpha
                m2_gemm(bc, xn, Wf(T.ssm.in_b), B, H, TSR, st);   // beta
            } else {
                gemv_batched_w(eng, ac, T.ssm.ref_in_a, xn, B, st);   // alpha
                gemv_batched_w(eng, bc, T.ssm.ref_in_b, xn, B, st);   // beta
            }
            qwen35_b_conv_kernel<<<dim3(grid1d((size_t)CONVD),B),256,0,st>>>(
                conv, qkv, eng->q35.ring[l], Wf(T.ssm.conv1d), d_slot, CONVD, B);
            qwen35_b_split_qkv_kernel<<<dim3(grid1d((size_t)CONVD),B),256,0,st>>>(qh, kh, vh, conv, B, CONVD, INNER);
            m2_l2norm_heads_kernel<<<dim3(NKH,B),128,0,st>>>(qh, NKH, SD, B);
            m2_l2norm_heads_kernel<<<dim3(NKH,B),128,0,st>>>(kh, NKH, SD, B);
            m2_decay_beta_kernel<<<grid1d((size_t)B*TSR),256,0,st>>>(gg, bb, ac, bc, Wf(T.ssm.a_log), Wf(T.ssm.dt_bias), B, TSR);
            qwen35_b_gdn_kernel<<<dim3(NVH,B),256,smGDN,st>>>(core, qh, kh, vh, gg, bb, eng->q35.S[l], d_slot, B, eng->format==FORMAT_FP8_BLOCK, NVH);  // 256 thr: 2x lanes on the SD*SD state loops
            m2_gated_norm_kernel<<<dim3(NVH,B),128,0,st>>>(gnorm, core, zc, Wf(T.ssm.norm), B, NVH, INNER);
            moe_profile_activation(eng, l, 1, gnorm, (size_t)B*INNER, st);
            gemv_batched_w(eng, mix, T.ssm.ref_out, gnorm, B, st);
        }
        residual_add_kernel<<<grid1d((size_t)B*H),256,0,st>>>(x, mix, B*H);
        rms_norm_rows_kernel<<<B,256,32*sizeof(float),st>>>(xn, x, Wf(T.ffn_norm), H, B, eps);
        moe_profile_activation(eng, l, 2, xn, (size_t)B*H, st);
        if (c->n_experts > 0) {   // Qwen3.5-MoE sparse block (experts + shared) → mix
            moe_ffn(eng, l, xn, mix, B, st);
        } else {
            gemv_batched_w(eng, fg, T.ref_gate, xn, B, st);
            gemv_batched_w(eng, fu, T.ref_up, xn, B, st);
            silu_glu_kernel<<<grid1d((size_t)B*I),256,0,st>>>(fa, fg, fu, B*I);
            gemv_batched_w(eng, mix, T.ref_down, fa, B, st);
        }
        residual_add_kernel<<<grid1d((size_t)B*H),256,0,st>>>(x, mix, B*H);
        q35_jspace_after_layer(eng, x, B, l, st);
        // S1a P4: capture this layer's residual (last row) into the draft aux-hidden concat when a
        // verify step is active and this target layer is configured. Gated + additive: a no-op
        // (single branch, no launch) when capture is inactive, so plain decode is byte-identical.
        if (eng->q35.dflash_capture_active && eng->q35.dflash_capture_layer[l]) {
            // Capture ALL B rows' residual at this configured target layer into the aux buffer,
            // laid out [feature_slot][row][H] so the drafter reads per-position aux (context + query
            // rows) as concat(F) per row. dflash_aux is sized F * capture_maxrows * H.
            int fslot = eng->q35.dflash_capture_slot[l];
            int maxr = eng->q35.dflash_capture_maxrows;
            int rr = (B < maxr) ? B : maxr;
            cudaMemcpyAsync(eng->q35.dflash_aux + (size_t)fslot * maxr * H,
                            x, (size_t)rr * H * sizeof(float),
                            cudaMemcpyDeviceToDevice, st);
        }
    }

    rms_norm_rows_kernel<<<B,256,32*sizeof(float),st>>>(xn, x, Wf(eng->tensors.output_norm), H, B, eps);
    if (eng->format == FORMAT_FP8_BLOCK && want_argmax && B == 1 && eng->d_lmhead_q8) {
        // EXACT two-pass greedy head at B=1: Q8_0 approx scan (0.53 GB, half the BF16 read) →
        // collect candidates within Q8HEAD_MARGIN of the approx max → exact BF16 rescore of
        // ≤Q8HEAD_MAXCAND rows → argmax. Bit-identical tokens (oracle + self-test gated).
        q8_head_gemv_kernel<<<(unsigned)((VOC + 7) / 8), 8 * 32, 0, st>>>(
            eng->d_sb[11], eng->d_lmhead_q8, xn, H, VOC);
        q8_head_candidates_kernel<<<1, 1024, 0, st>>>(eng->d_sb[11], VOC,
            eng->d_head_cand, eng->d_head_cnt);
        q8_head_rescore_argmax_kernel<<<1, 256, 0, st>>>(eng->d_lmhead_bf16, xn,
            eng->d_head_cand, eng->d_head_cnt, H, eng->d_ms_outtok);
        return;
    } else if (eng->format == FORMAT_FP8_BLOCK) {  // BF16 untied head (lossy heads flip the argmax);
        // weight-read-ONCE batched GEMV in ≤32-row chunks (P2): one pass reads the 2 GB head ONCE
        // for the whole decode batch (was ceil(B/16)× — 18.5 ms of a 91.6 ms B=30 step). The K≤32
        // kernel narrows the warp row-tile instead of splitting the batch; per-(token,row) k-order
        // is unchanged ⇒ logits bitwise-identical to the 16-row chunking this replaces.
        float *xt = workspace_data<float>(eng->q35.decode_workspace[Q35_QG]);
        // Per-layer scratch, free at head time; ≥ H·32 floats (Q35_QG row is 2·NQ·HD ≥ 2H, PF≥MS rows).
        for (int r0 = 0; r0 < B; r0 += 32) {
            int K = (B - r0 < 32) ? (B - r0) : 32;
            nvfp4_xT_launch(xt, xn + (size_t)r0 * H, H, K, st);
            bf16_head_gemv_batched_launch(eng->d_sb[11] + (size_t)r0 * VOC,
                                          eng->d_lmhead_bf16, xt, H, VOC, K, st);
        }
    } else
        gemv_batched_w(eng, eng->d_sb[11], Wq(eng->tensors.output_weight), xn, H, VOC, B, st, eng->tensors.output_fmt);
    if (want_argmax)
        argmax_rows_kernel<<<B,1024,0,st>>>(eng->d_sb[11], eng->d_ms_outtok, B, VOC);
    // S2a — writeback persistent per-slot state from the argmax output, device-side, so the NEXT
    // step's splice reads the token/position this step produced. In-graph on the greedy path
    // (want_argmax): the whole decode step is now a self-contained replayable graph. The sampled
    // path writes back in qwen35_ms_run after the sampler instead.
    if (splice && want_argmax)
        qwen35_writeback_slot_state_kernel<<<grid1d((size_t)B),256,0,st>>>(
            eng->q35.d_slot_tok, eng->q35.d_slot_pos, eng->d_ms_outtok, d_slot, B);
}

// ── S2b: keyed CUDA-graph cache ────────────────────────────────────────────────────────
// Key semantics (q35_make_decode_key / q35_graph_dominates) live in qwen35_graph_key.cuh.
// Look up a replayable graph for `want`. Dispatch is EXACT-match: the captured body runs
// exactly key.num_tokens rows of REAL work — fucina does not pad per-step inputs to a larger
// capture's row count — so a bigger graph cannot serve a smaller batch (it would process the
// extra rows at full cost; ~8× waste for a 31-row graph replaying a 4-row step). Dominance
// dispatch (q35_graph_dominates) is retained in the header as the primitive for the FUTURE
// S1/DFlash path, which WILL pad every device buffer to max shapes — only then does a larger
// capture correctly (and cheaply) serve a smaller uniform batch. Until that padding exists,
// exact-match is the only correct AND performant dispatch.
static cudaGraphExec_t q35_graph_lookup(gemma4_engine_t *eng, const q35_graph_key &want) {
    for (int i = 0; i < eng->q35.graph_count; i++) {
        const q35_graph_entry &e = eng->q35.graph_cache[i];
        if (e.exec && q35_graph_exact_match(e.key, want)) return e.exec;
    }
    return NULL;
}

// Capture the decode body for `key` once and instantiate it; replayed per step with the per-step
// varying device inputs refreshed outside the capture (tokens/positions/row→slot). num_tokens is
// the row count driven through qwen35_decode_multiseq_body (== num_reqs for plain decode).
static cudaGraphExec_t qwen35_ms_graph_ensure(gemma4_engine_t *eng, const q35_graph_key &key) {
    if (eng->fp4m_ssd_stream) return NULL; // SSD reads and staging copies are not graph-capturable
    if (key.num_tokens < 1 || key.num_reqs < 1 || key.num_reqs > eng->q35.capacity) return NULL;
    if (eng->q35.graph_failed) return NULL;
    if (cudaGraphExec_t hit = q35_graph_lookup(eng, key)) return hit;
    if (eng->q35.graph_count >= Q35_GRAPH_CACHE_CAP) return NULL; // cache full — per-kernel launches
    cudaStream_t cs = NULL; cudaGraph_t g = NULL; cudaGraphExec_t exec = NULL;
    int ok = cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) == cudaSuccess;
    if (ok && cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
        // Capture WITH the GPU input-splice + writeback so the graph is a self-contained decode
        // step: splice(state)→forward→argmax→writeback(state). Replay needs only the row→slot map.
        qwen35_decode_multiseq_body(eng, key.num_tokens, /*want_argmax=*/1,
                                    /*splice=*/eng->q35.gpu_splice_enabled, cs);
        ok = cudaStreamEndCapture(cs, &g) == cudaSuccess && g != NULL;
    } else ok = 0;
    if (ok) ok = cudaGraphInstantiate(&exec, g, 0) == cudaSuccess;
    if (g)  cudaGraphDestroy(g);
    if (cs) cudaStreamDestroy(cs);
    if (!ok || !exec) {
        eng->q35.graph_failed = 1; cudaGetLastError();
        fprintf(stderr, "fucina: qwen35 M4 batch graph capture failed "
                        "(nt=%d nr=%d utc=%d) — per-kernel launches\n",
                key.num_tokens, key.num_reqs, key.uniform_token_count);
        return NULL;
    }
    q35_graph_entry &slot = eng->q35.graph_cache[eng->q35.graph_count++];
    slot.key = key; slot.exec = exec;
    // graph_logged is a per-num_tokens dedupe bitmask (num_tokens<=64 fits Q35_GRAPH_CACHE_CAP).
    if (key.num_tokens < 64 && !(eng->q35.graph_logged & (1ULL << key.num_tokens))) {
        fprintf(stderr, "fucina: qwen35 M4 batch graph captured "
                        "(nt=%d nr=%d utc=%d)\n",
                key.num_tokens, key.num_reqs, key.uniform_token_count);
        eng->q35.graph_logged |= (1ULL << key.num_tokens);
    }
    return exec;
}

// Drop a graph from the cache (compacting swap-with-last) after a failed replay, so a later step
// can recapture it. Matches the OLD single-slot invalidate-on-replay-failure semantics.
static void q35_graph_evict(gemma4_engine_t *eng, cudaGraphExec_t exec) {
    for (int i = 0; i < eng->q35.graph_count; i++) {
        if (eng->q35.graph_cache[i].exec == exec) {
            cudaGraphExecDestroy(exec);
            eng->q35.graph_cache[i] = eng->q35.graph_cache[--eng->q35.graph_count];
            eng->q35.graph_cache[eng->q35.graph_count].exec = NULL;
            eng->q35.graph_cache[eng->q35.graph_count].key = q35_graph_key{0, 0, 0};
            return;
        }
    }
}

// Sample B Qwen rows with sequence-stable RNG. The generic sampler handles temp<=0 as argmax,
// so mixed greedy/sampled batches stay in one launch and each row is independent of batch order.
static void qwen35_sample_rows(gemma4_engine_t *eng, gemma4_seq **slv,
                               const float *logits, int B, cudaStream_t st) {
    float h_temp[GEMMA4_MAX_SEQS], h_topp[GEMMA4_MAX_SEQS];
    float h_minp[GEMMA4_MAX_SEQS], h_rnd[GEMMA4_MAX_SEQS];
    int h_topk[GEMMA4_MAX_SEQS];
    for (int r = 0; r < B; r++) {
        gemma4_seq *s = slv[r];
        h_temp[r] = s->samp_temp; h_topk[r] = s->samp_top_k;
        h_topp[r] = s->samp_top_p; h_minp[r] = s->samp_min_p;
        uint64_t z = (s->samp_seed ? s->samp_seed : 0x9e3779b97f4a7c15ULL)
                   + 0x9e3779b97f4a7c15ULL * (s->n_sampled + 1);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        h_rnd[r] = (float)((z >> 11) * (1.0 / 9007199254740992.0));
    }
    cudaMemcpyAsync(eng->d_ms_temp, h_temp, (size_t)B*sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(eng->d_ms_topk, h_topk, (size_t)B*sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(eng->d_ms_topp, h_topp, (size_t)B*sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(eng->d_ms_minp, h_minp, (size_t)B*sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(eng->d_ms_rnd, h_rnd, (size_t)B*sizeof(float), cudaMemcpyHostToDevice, st);
    sample_logits_ms_kernel<<<B,1024,0,st>>>(
        logits, eng->cfg.vocab_size, eng->d_ms_temp, eng->d_ms_topk, eng->d_ms_topp,
        eng->d_ms_minp, eng->d_ms_rnd, eng->d_ms_outtok);
}

// Refresh per-step inputs, then run one B-row forward. All-greedy batches use the captured graph;
// sampled or mixed batches use the same forward kernels followed by the on-device sampler.
// splice (S2a): when set, this step's input_ids/positions are derived ON-GPU from persistent
// per-slot state (d_slot_tok/d_slot_pos) — in_tok/positions are then ignored and NOT uploaded, and
// the sampled path's slot-state writeback happens here (the greedy path writes back in-graph).
// splice=0 keeps the legacy host-copy behavior (used by the sequential prefill path, whose
// per-position tokens are not in slot state). The row→slot map is always uploaded.
static int qwen35_ms_run(gemma4_engine_t *eng, gemma4_seq **slv, const int32_t *in_tok,
                         const int *positions, int B, int want_sample, int use_graph, int splice) {
    cudaStream_t st = eng->stream;
    int h_slot[GEMMA4_MAX_SEQS], any_sample = 0;
    for (int r = 0; r < B; r++) {
        h_slot[r] = (int)(slv[r] - eng->slots);
        if (slv[r]->samp_temp > 0.f) any_sample = 1;
    }
    if (!splice) {   // legacy: host supplies the input token + position for every row
        cudaMemcpyAsync((int32_t*)eng->d_sb[0], in_tok, (size_t)B*sizeof(int32_t), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(eng->d_ms_pos, positions, (size_t)B*sizeof(int), cudaMemcpyHostToDevice, st);
    }
    cudaMemcpyAsync(workspace_data<int>(eng->q35.routing_workspace),h_slot,
                    (size_t)B*sizeof(int),cudaMemcpyHostToDevice,st);
    if (use_graph && want_sample && !any_sample && eng->q35.graph_enabled) {
        q35_graph_key key = q35_make_decode_key(B);
        cudaGraphExec_t exec = qwen35_ms_graph_ensure(eng, key);
        if (exec) {
            if (cudaGraphLaunch(exec, st) == cudaSuccess) return 0;
            cudaGetLastError();
            q35_graph_evict(eng, exec); eng->q35.graph_failed = 1;
            fprintf(stderr, "fucina: qwen35 M4 graph replay failed — per-kernel launches\n");
        }
    }
    qwen35_decode_multiseq_body(eng, B, want_sample && !any_sample, splice, st);
    if (want_sample && any_sample) {
        qwen35_sample_rows(eng, slv, eng->d_sb[11], B, st);
        // Sampled path: body ran with want_argmax=0 (no in-graph writeback) — advance slot state
        // here from the sampler output so the next step's splice sees this step's token/position.
        if (splice)
            qwen35_writeback_slot_state_kernel<<<((unsigned)B+255)/256,256,0,st>>>(
                eng->q35.d_slot_tok, eng->q35.d_slot_pos, eng->d_ms_outtok,
                workspace_data<int>(eng->q35.routing_workspace), B);
    }
    return 0;
}

// Lazily (re)allocate the FULL-layer one-shot tensor-core attention scratch for a prompt of N
// rows. The O(NQ·N²) score/prob buffers dominate (~1.6 GB at N=4096), so this is built only on
// the first long base==0 prefill and grown as needed — never resident for short-prompt usage.
static bool ensure_q35_attn_scratch(gemma4_engine_t *eng, int N) {
    if (N <= eng->q35.attn_cap && eng->q35.scores) return true;
    const int NQ = eng->cfg.n_heads, NKV = eng->cfg.n_kv_global, HD = M2_HEAD;
    auto bytes_for = [&](int cap) -> size_t {
        if (cap <= 0) return 0;
        size_t oqc=(size_t)NQ*HD, okvc=(size_t)NKV*HD, nnc=(size_t)NQ*cap*cap;
        return (size_t)cap*(3*oqc+2*okvc)*sizeof(__nv_bfloat16)
             + nnc*(sizeof(__nv_bfloat16)+sizeof(float));
    };
    size_t old_bytes = bytes_for(eng->q35.attn_cap), new_bytes = bytes_for(N);
    if (new_bytes > old_bytes && !q35_can_grow(eng, new_bytes - old_bytes)) return false;
    q35_account_sub(eng, old_bytes);
    #define Q35_FREE(p) do { if (p) { cudaFree(p); (p) = NULL; } } while (0)
    Q35_FREE(eng->q35.qb); Q35_FREE(eng->q35.kb); Q35_FREE(eng->q35.vb);
    Q35_FREE(eng->q35.kbx); Q35_FREE(eng->q35.vbx); Q35_FREE(eng->q35.pb);
    Q35_FREE(eng->q35.scores);
    eng->q35.attn_cap = 0;
    const size_t oq = (size_t)NQ * HD, okv = (size_t)NKV * HD, nn = (size_t)NQ * N * N;
    bool ok = true;
    ok &= cudaMalloc(&eng->q35.qb,  (size_t)N*oq*sizeof(__nv_bfloat16)) == cudaSuccess;
    ok &= cudaMalloc(&eng->q35.kb,  (size_t)N*okv*sizeof(__nv_bfloat16)) == cudaSuccess;
    ok &= cudaMalloc(&eng->q35.vb,  (size_t)N*okv*sizeof(__nv_bfloat16)) == cudaSuccess;
    ok &= cudaMalloc(&eng->q35.kbx, (size_t)N*oq*sizeof(__nv_bfloat16)) == cudaSuccess;
    ok &= cudaMalloc(&eng->q35.vbx, (size_t)N*oq*sizeof(__nv_bfloat16)) == cudaSuccess;
    ok &= cudaMalloc(&eng->q35.pb,  nn*sizeof(__nv_bfloat16)) == cudaSuccess;
    ok &= cudaMalloc(&eng->q35.scores, nn*sizeof(float)) == cudaSuccess;
    if (!ok) {
        Q35_FREE(eng->q35.qb); Q35_FREE(eng->q35.kb); Q35_FREE(eng->q35.vb);
        Q35_FREE(eng->q35.kbx); Q35_FREE(eng->q35.vbx); Q35_FREE(eng->q35.pb);
        Q35_FREE(eng->q35.scores);
        return false;
    }
    eng->q35.attn_cap = N;
    q35_account_add(eng, bytes_for(N));
    return true;
    #undef Q35_FREE
}

// FULL-layer one-shot prefill attention via tensor-core GEMMs (base==0: queries attend only
// within the tile, which IS the whole causal prompt). Replaces the scalar O(N²) qwen35_b_attn
// kernel. qb is post q-norm + RoPE [N][NQ][HD]; kb/vb post k-norm+RoPE / raw value [N][NKV][HD].
// Writes attn[N][NQ*HD]. Scale rsqrt(HD), causal, no softcap — identical math to the scalar path.
static void q35_full_attn_tc(gemma4_engine_t *eng, float *attn, const float *qb,
                             const float *kb, const float *vb, int N, cudaStream_t st) {
    const int NQ = eng->cfg.n_heads, NKV = eng->cfg.n_kv_global, HD = M2_HEAD;
    const size_t oq = (size_t)NQ * HD, okv = (size_t)NKV * HD;
    auto g1 = [](size_t n){ return (unsigned)((n + 255) / 256); };
    f32_to_bf16_kernel<<<g1((size_t)N*oq),256,0,st>>>(eng->q35.qb, qb, (size_t)N*oq);
    f32_to_bf16_kernel<<<g1((size_t)N*okv),256,0,st>>>(eng->q35.kb, kb, (size_t)N*okv);
    f32_to_bf16_kernel<<<g1((size_t)N*okv),256,0,st>>>(eng->q35.vb, vb, (size_t)N*okv);
    kv_broadcast_bf16_kernel<<<dim3(g1(HD),NQ,N),256,0,st>>>(eng->q35.kbx, eng->q35.kb, N, NQ, NKV, HD);
    kv_broadcast_bf16_kernel<<<dim3(g1(HD),NQ,N),256,0,st>>>(eng->q35.vbx, eng->q35.vb, N, NQ, NKV, HD);
    const float scale = rsqrtf((float)HD), b0 = 0.0f;
    long long sNN = (long long)N * N;
    // S[h] = scale · Q[h]ᵀ·K[h]   (col-major [hd×N], ld=oq, stride=hd)
    cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, N, HD,
        &scale, eng->q35.qb,  CUDA_R_16BF, (int)oq, (long long)HD,
                eng->q35.kbx, CUDA_R_16BF, (int)oq, (long long)HD,
        &b0,    eng->q35.scores, CUDA_R_32F, N, sNN,
        NQ, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    attn_softmax_batched_kernel<<<dim3(N,NQ),256,32*sizeof(float),st>>>(eng->q35.scores, eng->q35.pb, N, 0);
    const float a1 = 1.0f;
    // O[h] = V[h]·P[h]ᵀ  (col-major [hd×N], ld=oq, stride=hd)
    cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_N, CUBLAS_OP_T, HD, N, N,
        &a1, eng->q35.vbx, CUDA_R_16BF, (int)oq, (long long)HD,
             eng->q35.pb,  CUDA_R_16BF, N, sNN,
        &b0, attn, CUDA_R_32F, (int)oq, (long long)HD,
        NQ, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Lazily (re)allocate the base>0 CONTINUATION tensor-core attention scratch: an fp16 Q tile of
// `qelems` plus fp32 score + fp16 prob buffers of `elems` each, grown to the largest request
// seen (never sized up-front — the box is memory-pressured). Returns false on alloc failure;
// the caller then falls back to the scalar kernel for THIS chunk and retries next time
// (same retry semantics as ensure_q35_attn_scratch — never disabled, never crashes).
static bool ensure_q35_cont_scratch(gemma4_engine_t *eng, size_t qelems, size_t elems) {
    if (qelems > eng->q35.qh_cap || !eng->q35.qh) {
        size_t oldb=eng->q35.qh_cap*sizeof(__half), newb=qelems*sizeof(__half);
        if (newb > oldb && !q35_can_grow(eng, newb-oldb)) return false;
        q35_account_sub(eng, oldb);
        if (eng->q35.qh) { cudaFree(eng->q35.qh); eng->q35.qh = NULL; }
        eng->q35.qh_cap = 0;
        if (cudaMalloc(&eng->q35.qh, qelems * sizeof(__half)) != cudaSuccess) {
            eng->q35.qh = NULL;
            return false;
        }
        eng->q35.qh_cap = qelems;
        q35_account_add(eng, qelems * sizeof(__half));
    }
    if (elems > eng->q35.cont_cap || !eng->q35.cont_scores || !eng->q35.cont_p) {
        size_t oldb=eng->q35.cont_cap*(sizeof(float)+sizeof(__half));
        size_t newb=elems*(sizeof(float)+sizeof(__half));
        if (newb > oldb && !q35_can_grow(eng, newb-oldb)) return false;
        q35_account_sub(eng, oldb);
        if (eng->q35.cont_scores) { cudaFree(eng->q35.cont_scores); eng->q35.cont_scores = NULL; }
        if (eng->q35.cont_p)      { cudaFree(eng->q35.cont_p);      eng->q35.cont_p = NULL; }
        eng->q35.cont_cap = 0;
        bool ok = cudaMalloc(&eng->q35.cont_scores, elems * sizeof(float)) == cudaSuccess;
        ok = ok && cudaMalloc(&eng->q35.cont_p, elems * sizeof(__half)) == cudaSuccess;
        if (!ok) {
            if (eng->q35.cont_scores) { cudaFree(eng->q35.cont_scores); eng->q35.cont_scores = NULL; }
            if (eng->q35.cont_p)      { cudaFree(eng->q35.cont_p);      eng->q35.cont_p = NULL; }
            return false;
        }
        eng->q35.cont_cap = elems;
        q35_account_add(eng, elems * (sizeof(float) + sizeof(__half)));
    }
    return true;
}

// FULL-layer CONTINUATION attention (base>0 chunked prefill) via tensor-core GEMMs reading the
// fp16 K/V cache IN PLACE — replaces the scalar qwen35_b_attn_kernel walk of the whole cache
// (measured 38–44× at this geometry: 82→2.1 ms/layer at S=2048, 426→9.7 ms at S=8192). Per KV
// group g one strided-batched fp16×fp16→fp32 QKᵀ GEMM broadcasts the cached K over the group's
// NQ/NKV query heads with strideA=0 (no kv-broadcast copies, no bf16 cache conversion), then the
// KEY-MAJOR rectangular causal softmax, then the same strided-batched V·P GEMM. Query rows are
// sub-tiled to bound the score scratch at ~96M elements (384 MB fp32 + 192 MB fp16). The current
// chunk's K/V are already in the cache (qwen35_b_kv_write_kernel runs before the attention
// branch). Returns false on any alloc/cuBLAS failure — attn may then be partially written, which
// is safe because the scalar fallback rewrites ALL rows. NOT bitwise vs the scalar kernel
// (fp16 Q + tensor-core reduction order); token-level parity is the gate, matching the TC
// base==0 precedent — chunked prefill already mixes TC (chunk 1) with this path (chunk 2+).
static bool q35_cont_attn_tc(gemma4_engine_t *eng, float *attn, const float *qb,
                             int l, int slot, int base, int T, cudaStream_t st) {
    const int NQ = eng->cfg.n_heads, NKV = eng->cfg.n_kv_global, HD = M2_HEAD;
    if (NKV <= 0 || NQ % NKV != 0 || !eng->q35.Kc_slot[l][slot] ||
        !eng->q35.Vc_slot[l][slot]) return false;
    const int G = NQ / NKV, S = base + T;
    const __half *Kc = eng->q35.Kc_slot[l][slot];   // [pos][NKV][HD]
    const __half *Vc = eng->q35.Vc_slot[l][slot];
    int rows_sub = (int)(((size_t)96 << 20) / ((size_t)NQ * S));   // NQ*rows_sub*S <= 96M elems
    if (rows_sub < 64) rows_sub = 64;
    if (rows_sub > T) rows_sub = T;
    if (!ensure_q35_cont_scratch(eng, (size_t)T * NQ * HD, (size_t)NQ * rows_sub * S))
        return false;
    f32_to_f16_kernel<<<(unsigned)(((size_t)T * NQ * HD + 255) / 256), 256, 0, st>>>(
        eng->q35.qh, qb, (size_t)T * NQ * HD);
    const float scale = rsqrtf((float)HD), a1 = 1.0f, b0 = 0.0f;
    for (int t0 = 0; t0 < T; t0 += rows_sub) {
        const int Tq = (T - t0 < rows_sub) ? T - t0 : rows_sub;
        for (int g = 0; g < NKV; g++) {
            // scores[h] (S×Tq col-major, ld=S) = scale · K_gᵀ(S×HD) · Q_h(HD×Tq); the cached
            // K_g (lda=NKV·HD over positions) is broadcast across the group via strideA=0.
            if (cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_T, CUBLAS_OP_N, S, Tq, HD,
                    &scale,
                    Kc + (size_t)g * HD, CUDA_R_16F, NKV * HD, 0,
                    eng->q35.qh + ((size_t)t0 * NQ + (size_t)g * G) * HD, CUDA_R_16F,
                        NQ * HD, (long long)HD,
                    &b0, eng->q35.cont_scores + (size_t)g * G * S * Tq, CUDA_R_32F,
                        S, (long long)S * Tq,
                    G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
                return false;
        }
        attn_softmax_rect_kernel<<<dim3(Tq, NQ), 256, 32 * sizeof(float), st>>>(
            eng->q35.cont_scores, eng->q35.cont_p, Tq, S, base + t0);
        for (int g = 0; g < NKV; g++) {
            // O_h (HD×Tq, into attn row-major [T][NQ][HD]) = V_g (HD×S, strideA=0) · P_h (S×Tq)
            if (cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_N, CUBLAS_OP_N, HD, Tq, S,
                    &a1,
                    Vc + (size_t)g * HD, CUDA_R_16F, NKV * HD, 0,
                    eng->q35.cont_p + (size_t)g * G * S * Tq, CUDA_R_16F, S, (long long)S * Tq,
                    &b0, attn + ((size_t)t0 * NQ + (size_t)g * G) * HD, CUDA_R_32F,
                        NQ * HD, (long long)HD,
                    G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
                return false;
        }
    }
    return true;
}

// qwen35 prefill weight-cache slots (one per distinct projection a layer can hold).
enum { WC_QKV=0, WC_Z, WC_OUT, WC_Q, WC_K, WC_V, WC_O, WC_GATE, WC_UP, WC_DOWN, WC_A, WC_B };

// Tensor-core BF16 GEMM for ONE qwen35 prefill projection: get the BF16 weight (from the resident
// cache, or dequant on the fly), cast the FP32 activation tile to BF16, then a cuBLAS tensor-core
// GEMM → FP32 dst[T×out_dim]. Prefill is COMPUTE-bound, so this replaces the decode-oriented dp4a
// gemv_batched_w (warp-per-output-row, ~3 effective TFLOPS) built for 1–few rows, not a wide tile.
// dst/x token-major; weight row-major [out_dim][in_dim] (= BF16 col-major [in_dim×out_dim]).
// wcslot: address of this weight's cache entry (NULL = never cache).
// One-time NVFP4 machinery setup for the qwen35 hybrid (build_fp4_weights is gated off for it):
// the shared cuBLASLt handle + fp4_desc (per-16 E4M3 block scales, device pointer-mode) +
// activation scalars/workspace + the per-(layer,slot) global-scale store. Env FUCINA_QWEN35_FP4
// (default on): NVFP4 is 4-bit (~3.5 GB for 9B) — smaller than the BF16 cache AND ~2× FP8 GEMM.
static int q35_fp4_setup(gemma4_engine_t *eng) {
    if (eng->q35.fp4_on) return 0;
    // OPT-IN (FUCINA_QWEN35_FP4=1): NVFP4 is 4-bit — matches the BF16 oracle on real text (8/8) and
    // is much faster + smaller, but a random-token stress prompt can flip an argmax near-tie, so the
    // strict slow==fast self-consistency gate isn't guaranteed. Default keeps the BF16 cache path.
    const char *e = getenv("FUCINA_QWEN35_FP4");
    if (!e || atoi(e) == 0) {
        fprintf(stderr, "fucina: qwen35 NVFP4 prefill GEMM off (not requested; FUCINA_QWEN35_FP4!=1)\n");
        return -1;
    }
    fprintf(stderr, "fucina: qwen35 NVFP4 prefill GEMM setup attempting\n");
    if (!eng->cublaslt && cublasLtCreate(&eng->cublaslt) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "fucina: qwen35 NVFP4 prefill GEMM setup failed (cublasLtCreate)\n");
        return -1;
    }
    const int L = eng->cfg.n_layers;
    bool ok = true;
    if (!eng->q35.fp4_gsw) ok &= cudaMalloc(&eng->q35.fp4_gsw, (size_t)L*12*sizeof(float)) == cudaSuccess;
    if (!eng->d_fp4_amax)  ok &= cudaMalloc(&eng->d_fp4_amax,  sizeof(float))    == cudaSuccess;
    if (!eng->d_fp4_gsact) ok &= cudaMalloc(&eng->d_fp4_gsact, sizeof(float))    == cudaSuccess;
    if (!eng->d_fp4_alpha) ok &= cudaMalloc(&eng->d_fp4_alpha, 2*sizeof(float))  == cudaSuccess;
    if (!eng->d_fp4_ws)    ok &= cudaMalloc(&eng->d_fp4_ws,    64ull<<20)        == cudaSuccess;
    if (ok && !eng->fp4_desc) {
        cublasLtMatmulDescCreate(&eng->fp4_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        int32_t smode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &smode, sizeof(smode));
        cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &smode, sizeof(smode));
        int32_t pmode=CUBLASLT_POINTER_MODE_DEVICE;
        cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pmode, sizeof(pmode));
    }
    if (!ok) {
        fprintf(stderr, "fucina: qwen35 NVFP4 prefill GEMM setup failed (device allocation/descriptor)\n");
        return -1;
    }
    eng->q35.fp4_on = 1;
    fprintf(stderr, "fucina: qwen35 NVFP4 prefill GEMM ON (4-bit tensor cores, ~2x FP8)\n");
    return 0;
}

// Tensor-core prefill projection: dst[T×out_dim] = X[T×in_dim] @ Wᵀ. Prefers NVFP4 (4-bit tensor
// cores, ~2x FP8) with lazily-built resident weights; falls back to the BF16 weight-cache, else a
// per-tile dequant. l/slot index the per-projection weight stores (d_q35_fp4_w / d_q35_wc).
static void q35_proj_gemm(gemma4_engine_t *eng, float *dst, uint64_t off, int fmt,
                          const float *x_f32, int in_dim, int out_dim, int T, cudaStream_t st,
                          int l, int slot, const WeightRef *ref = nullptr) {
    const uint8_t *w = ref ? ref->data : weight_fp8(eng, off);
    if (ref) { fmt=weight_ref_format(*ref); in_dim=ref->in_dim; out_dim=ref->out_dim; }
    WeightRef &wc_ref=eng->q35.wc_ref[l][slot], &fp4_ref=eng->q35.fp4_ref[l][slot];
    __nv_bfloat16 *weight_workspace[2]={
        (__nv_bfloat16*)eng->q35.prefill_weight_workspace[0].data,
        (__nv_bfloat16*)eng->q35.prefill_weight_workspace[1].data};
    __nv_bfloat16 *input_workspace=(__nv_bfloat16*)eng->q35.prefill_input_workspace.data;
    auto bind_variant=[&](WeightRef &variant,const void *data,const void *scale,const float *global,
                          WeightEncoding encoding,TensorLayout layout,uint16_t flags){
        variant.data=(const uint8_t*)data; variant.scale=scale; variant.global_scale=global;
        variant.out_dim=out_dim; variant.in_dim=in_dim; variant.encoding=encoding;
        variant.layout=layout; variant.flags=flags;
    };
    // Qwen3.5 block-FP8. Short tiles (≤2 GEMV chunks): the float-activation FP8 GEMM, weight
    // amortized across ≤FP8_MAXB rows. WIDE prefill tiles: dequant the projection to BF16 ONCE
    // (cached in the per-layer weight-cache slot when memory allows) and ride the tensor-core
    // BF16 GEMM — the chunked GEMV re-read the whole weight T/FP8_MAXB times per tile, which
    // was the dominant prefill cost (43.9 s TTFT at a ~2k prompt on the 35B MoE).
    // P1 unification: the short-tile fp8_block path and the wide-tile bf16-dequant path are
    // DIFFERENT numerics (fp8 weights in the GEMM vs bf16-dequanted weights). A short prompt
    // therefore computes differently standalone (fp8_block) than inside a batched multi-seq
    // prefill (bf16, large Ttot) — breaking batched==standalone token-equality. FUCINA_UNIFY_BF16
    // forces every prefill projection onto the bf16 path so standalone and batched match (a
    // deliberate, strictly-more-accurate numerics change; measure the short-prompt TTFT cost).
    static int unify_bf16 = -1;
    if (unify_bf16 < 0) unify_bf16 = (getenv("FUCINA_NO_UNIFY") != NULL) ? 0 : 1;   // default ON (deterministic keeper)
    if (fmt == FORMAT_FP8_BLOCK) {
        const __nv_bfloat16 *sc = ref ? (const __nv_bfloat16*)ref->scale : wscale_fp8(eng, w);
        if (!unify_bf16 && T <= 2 * FP8_MAXB) {
            for (int b0 = 0; b0 < T; b0 += FP8_MAXB) {
                int bb = (T - b0 < FP8_MAXB) ? (T - b0) : FP8_MAXB;
                fp8_block_gemm_launch(dst + (size_t)b0*out_dim, w, sc, x_f32 + (size_t)b0*in_dim,
                                      in_dim, out_dim, bb, st);
            }
            return;
        }
        const uint64_t nfp = (uint64_t)in_dim * out_dim;
        f32_to_bf16_kernel<<<(unsigned)(((size_t)T * in_dim + 255) / 256), 256, 0, st>>>(
            input_workspace, x_f32, (uint64_t)T * in_dim);
        __nv_bfloat16 **wcslot = &eng->q35.wc[l][slot];
        const __nv_bfloat16 *wbf;
        if (wc_ref.data) {
            wbf = (const __nv_bfloat16*)wc_ref.data;
        } else if (eng->q35.wcache_on) {
            __nv_bfloat16 *buf = NULL;
            if (cudaMalloc(&buf, nfp * sizeof(__nv_bfloat16)) == cudaSuccess) {
                dequant_fp8_block_to_bf16_kernel<<<(unsigned)((nfp + 255) / 256), 256, 0, st>>>(
                    buf, w, sc, in_dim, nfp);
                *wcslot = buf; wbf = buf;
                bind_variant(wc_ref,buf,nullptr,nullptr,WeightEncoding::BF16,TensorLayout::ROW_MAJOR,
                             WEIGHT_FLAG_CACHE);
                q35_account_add(eng, nfp * sizeof(__nv_bfloat16));
            } else {
                cudaGetLastError(); eng->q35.wcache_on = 0;
                dequant_fp8_block_to_bf16_kernel<<<(unsigned)((nfp + 255) / 256), 256, 0, st>>>(
                    weight_workspace[0], w, sc, in_dim, nfp);
                wbf = weight_workspace[0];
            }
        } else {
            dequant_fp8_block_to_bf16_kernel<<<(unsigned)((nfp + 255) / 256), 256, 0, st>>>(
                weight_workspace[0], w, sc, in_dim, nfp);
            wbf = weight_workspace[0];
        }
        gemm_bf16(eng, wbf, input_workspace, dst, in_dim, out_dim, T);
        return;
    }
    bool packed = ref ? ref->layout==TensorLayout::Q4K_PACKED : use_packed_q4k(eng, fmt, w);
    const uint64_t n = (uint64_t)in_dim * out_dim;
    // activation → BF16 once (both the NVFP4 and BF16 GEMM paths consume it)
    f32_to_bf16_kernel<<<(unsigned)(((size_t)T * in_dim + 255) / 256), 256, 0, st>>>(
        input_workspace, x_f32, (uint64_t)T * in_dim);
    // ── NVFP4 path (4-bit tensor cores) ──
    if (eng->q35.fp4_on) {
        uint8_t **wslot = &eng->q35.fp4_w[l][slot], **wscslot = &eng->q35.fp4_wsc[l][slot];
        if (!fp4_ref.data) {                                   // first touch: dequant→BF16→NVFP4-quantize once
            size_t pk = (size_t)out_dim * (in_dim/2);
            size_t sw = (size_t)nvfp4_pad(out_dim,128) * nvfp4_pad(in_dim/NVFP4_BLK,4);
            if (cudaMalloc(wslot, pk) == cudaSuccess && cudaMalloc(wscslot, sw) == cudaSuccess) {
                cudaMemsetAsync(*wscslot, 0, sw, st);
                dequant_proj_to_bf16(weight_workspace[0], w, n, fmt, packed, st);
                nvfp4_quantize(weight_workspace[0], out_dim, in_dim, *wslot,
                               (uint8_t*)weight_workspace[1], *wscslot,
                               eng->q35.fp4_gsw + (l*12+slot), eng->d_fp4_amax, st);
                bind_variant(fp4_ref,*wslot,*wscslot,eng->q35.fp4_gsw+(l*12+slot),
                             WeightEncoding::NVFP4_SWIZZLED,TensorLayout::NVFP4_SCALE_SWIZZLED,
                             WEIGHT_FLAG_CACHE|WEIGHT_FLAG_PACKED);
                q35_account_add(eng, pk + sw);
            } else {
                if (*wslot) { cudaFree(*wslot); *wslot=NULL; }
                if (*wscslot) { cudaFree(*wscslot); *wscslot=NULL; }
                fp4_ref={}; eng->q35.fp4_on = 0;                     // OOM → fall back to BF16 cache
            }
        }
        if (fp4_ref.data && gemm_nvfp4_q35(eng, fp4_ref.data,(const uint8_t*)fp4_ref.scale,
                                     fp4_ref.global_scale,input_workspace,dst,in_dim,out_dim,T,st))
            return;
    }
    // ── BF16 weight-cache fallback ──
    __nv_bfloat16 **wcslot = &eng->q35.wc[l][slot];
    const __nv_bfloat16 *wbf;
    if (wc_ref.data) {
        wbf = (const __nv_bfloat16*)wc_ref.data;
    } else if (eng->q35.wcache_on) {
        __nv_bfloat16 *buf = NULL;
        if (cudaMalloc(&buf, n * sizeof(__nv_bfloat16)) == cudaSuccess) {
            dequant_proj_to_bf16(buf, w, n, fmt, packed, st);
            *wcslot = buf; wbf = buf;
            bind_variant(wc_ref,buf,nullptr,nullptr,WeightEncoding::BF16,TensorLayout::ROW_MAJOR,
                         WEIGHT_FLAG_CACHE);
            q35_account_add(eng, n * sizeof(__nv_bfloat16));
        } else {
            eng->q35.wcache_on = 0;
            dequant_proj_to_bf16(weight_workspace[0], w, n, fmt, packed, st);
            wbf = weight_workspace[0];
        }
    } else {
        dequant_proj_to_bf16(weight_workspace[0], w, n, fmt, packed, st);
        wbf = weight_workspace[0];
    }
    gemm_bf16(eng, wbf, input_workspace, dst, in_dim, out_dim, T);
}

// Materialize one slot's hybrid state as an all-or-nothing transaction. Allocations are pooled:
// seq_remove marks the slot free but retains its state, so churn never enters cudaMalloc/cudaFree.
// Fixed GDN state + conv history is one aligned slab; the per-layer pointers are non-owning views.
static int q35_slot_state_ensure(gemma4_engine_t *eng, int slot) {
    if (slot < 0 || slot >= eng->q35.capacity) return -1;
    if (eng->q35.slot_allocated[slot]) return 0;
    const gemma4_model_config_t *c=&eng->cfg;
    const int NKV=c->n_kv_global, HD=M2_HEAD, NVH=c->ssm_inner_size/M2_SD;
    const int CONVD=2*M2_KEYD+c->ssm_inner_size, maxctx=eng->q35.maxctx;
    const int initial_kv = maxctx < 256 ? maxctx : 256;
    const size_t initial_kv_bytes = eng->q35.per_slot_kv_bytes * (size_t)initial_kv / maxctx;
    if (!q35_can_grow(eng, eng->q35.per_slot_recurrent_bytes + initial_kv_bytes)) return -1;
    bool ok = cudaMalloc(&eng->q35.recurrent_slab[slot],
                         eng->q35.per_slot_recurrent_bytes) == cudaSuccess;
    size_t recurrent_off = 0;
    const size_t s_bytes = (size_t)NVH*M2_SD*M2_SD*sizeof(__nv_bfloat16);
    const size_t ring_bytes = (size_t)CONVD*(M2_CK-1)*sizeof(float);
    for (int l=0; l<c->n_layers && ok; l++) {
        if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
            ok &= cudaMalloc(&eng->q35.Kc_slot[l][slot],
                             (size_t)initial_kv*NKV*HD*sizeof(__half)) == cudaSuccess;
            ok &= cudaMalloc(&eng->q35.Vc_slot[l][slot],
                             (size_t)initial_kv*NKV*HD*sizeof(__half)) == cudaSuccess;
        } else {
            recurrent_off = q35_align_up(recurrent_off);
            eng->q35.S_slot[l][slot] = (__nv_bfloat16 *)(eng->q35.recurrent_slab[slot] + recurrent_off);
            recurrent_off += s_bytes;
            recurrent_off = q35_align_up(recurrent_off);
            eng->q35.ring_slot[l][slot] = (float *)(eng->q35.recurrent_slab[slot] + recurrent_off);
            recurrent_off += ring_bytes;
        }
    }
    if (q35_align_up(recurrent_off) != eng->q35.per_slot_recurrent_bytes) ok = false;
    if (!ok) {
        if (eng->q35.recurrent_slab[slot]) cudaFree(eng->q35.recurrent_slab[slot]);
        eng->q35.recurrent_slab[slot] = NULL;
        for (int l=0; l<c->n_layers; l++) {
            if (eng->q35.Kc_slot[l][slot]) cudaFree(eng->q35.Kc_slot[l][slot]);
            if (eng->q35.Vc_slot[l][slot]) cudaFree(eng->q35.Vc_slot[l][slot]);
            eng->q35.S_slot[l][slot]=NULL; eng->q35.ring_slot[l][slot]=NULL;
            eng->q35.Kc_slot[l][slot]=NULL; eng->q35.Vc_slot[l][slot]=NULL;
        }
        cudaGetLastError();
        return -1;
    }
    bind_workspace(eng->q35.recurrent_workspace[slot],eng->q35.recurrent_slab[slot],
                   eng->q35.per_slot_recurrent_bytes,WorkspaceKind::RECURRENT_STATE);
    const size_t kv_layer_bytes=(size_t)initial_kv*NKV*HD*sizeof(__half);
    for(int l=0;l<c->n_layers;l++) if(c->attn_kind[l]==GEMMA4_ATTN_FULL){
        bind_workspace(eng->q35.kv_key_workspace[l][slot],eng->q35.Kc_slot[l][slot],
                       kv_layer_bytes,WorkspaceKind::KV_CACHE);
        bind_workspace(eng->q35.kv_value_workspace[l][slot],eng->q35.Vc_slot[l][slot],
                       kv_layer_bytes,WorkspaceKind::KV_CACHE);
    }
    // Publish only after every allocation succeeded; graph kernels can never observe a partial slot.
    cudaStream_t st=eng->stream;
    for (int l=0; l<c->n_layers; l++) {
        if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
            cudaMemcpyAsync(eng->q35.Kc[l]+slot, &eng->q35.Kc_slot[l][slot],
                            sizeof(__half*), cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(eng->q35.Vc[l]+slot, &eng->q35.Vc_slot[l][slot],
                            sizeof(__half*), cudaMemcpyHostToDevice, st);
        } else {
            cudaMemcpyAsync(eng->q35.S[l]+slot, &eng->q35.S_slot[l][slot],
                            sizeof(__nv_bfloat16*), cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(eng->q35.ring[l]+slot, &eng->q35.ring_slot[l][slot],
                            sizeof(float*), cudaMemcpyHostToDevice, st);
        }
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) return -1;
    eng->q35.slot_allocated[slot]=1; eng->q35.kv_capacity[slot]=initial_kv;
    eng->q35.allocated_slots++;
    q35_account_add(eng, eng->q35.per_slot_recurrent_bytes + initial_kv_bytes);
    return 0;
}

// Grow one slot's FULL-layer KV geometrically in 256-token blocks. Device pointer tables let us
// replace allocations without invalidating captured graphs. Existing prefixes are copied before
// publishing the new pointers; old blocks are freed only after publication is stream-complete.
static int q35_slot_kv_reserve(gemma4_engine_t *eng, int slot, int need_tokens) {
    if (slot < 0 || slot >= eng->q35.capacity || !eng->q35.slot_allocated[slot]) return -1;
    int oldcap=eng->q35.kv_capacity[slot];
    if (need_tokens <= oldcap) return 0;
    if (need_tokens > eng->q35.maxctx) return -1;
    int newcap=oldcap > 0 ? oldcap*2 : 256;
    if (newcap < need_tokens) newcap=need_tokens;
    newcap=(newcap+255)&~255;
    if (newcap > eng->q35.maxctx) newcap=eng->q35.maxctx;
    const gemma4_model_config_t *c=&eng->cfg;
    const int NKV=c->n_kv_global, HD=M2_HEAD;
    const size_t old_bytes=eng->q35.per_slot_kv_bytes*(size_t)oldcap/eng->q35.maxctx;
    const size_t new_bytes=eng->q35.per_slot_kv_bytes*(size_t)newcap/eng->q35.maxctx;
    // During migration both generations coexist, so require room for the full new generation.
    if (!q35_can_grow(eng, new_bytes)) return -1;
    __half *newK[GEMMA4_CAP_LAYERS]={}, *newV[GEMMA4_CAP_LAYERS]={};
    bool ok=true;
    for (int l=0; l<c->n_layers && ok; l++) if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
        ok &= cudaMalloc(&newK[l], (size_t)newcap*NKV*HD*sizeof(__half)) == cudaSuccess;
        ok &= cudaMalloc(&newV[l], (size_t)newcap*NKV*HD*sizeof(__half)) == cudaSuccess;
    }
    if (!ok) {
        for (int l=0; l<c->n_layers; l++) { if (newK[l]) cudaFree(newK[l]); if (newV[l]) cudaFree(newV[l]); }
        cudaGetLastError(); return -1;
    }
    size_t transient=eng->q35.committed_bytes+new_bytes;
    if (transient > eng->q35.peak_bytes) eng->q35.peak_bytes=transient;
    cudaStream_t st=eng->stream;
    const size_t copy_bytes=(size_t)oldcap*NKV*HD*sizeof(__half);
    for (int l=0; l<c->n_layers; l++) if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
        if (oldcap) {
            cudaMemcpyAsync(newK[l], eng->q35.Kc_slot[l][slot], copy_bytes,
                            cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(newV[l], eng->q35.Vc_slot[l][slot], copy_bytes,
                            cudaMemcpyDeviceToDevice, st);
        }
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) return -1;
    for (int l=0; l<c->n_layers; l++) if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
        cudaMemcpyAsync(eng->q35.Kc[l]+slot, &newK[l], sizeof(__half*), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(eng->q35.Vc[l]+slot, &newV[l], sizeof(__half*), cudaMemcpyHostToDevice, st);
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) return -1;
    for (int l=0; l<c->n_layers; l++) if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
        cudaFree(eng->q35.Kc_slot[l][slot]); cudaFree(eng->q35.Vc_slot[l][slot]);
        eng->q35.Kc_slot[l][slot]=newK[l]; eng->q35.Vc_slot[l][slot]=newV[l];
        const size_t layer_bytes=(size_t)newcap*NKV*HD*sizeof(__half);
        bind_workspace(eng->q35.kv_key_workspace[l][slot],newK[l],layer_bytes,WorkspaceKind::KV_CACHE);
        bind_workspace(eng->q35.kv_value_workspace[l][slot],newV[l],layer_bytes,WorkspaceKind::KV_CACHE);
    }
    eng->q35.kv_capacity[slot]=newcap;
    q35_account_add(eng, new_bytes-old_bytes);
    return 0;
}

// Zero a slot's GDN recurrent state + conv ring (the FULL-layer K/V cache needs no zeroing —
// attention only reads positions [0..pos] written this sequence).
static void qwen35_slot_reset(gemma4_engine_t *eng, int slot, cudaStream_t st) {
    const gemma4_model_config_t *c = &eng->cfg;
    const int NVH=(c->ssm_inner_size/M2_SD), SD=M2_SD, CONVD=(2*M2_KEYD+c->ssm_inner_size), CK=M2_CK;
    for (int l = 0; l < c->n_layers; l++) if (c->attn_kind[l] != GEMMA4_ATTN_FULL) {
        cudaMemsetAsync(eng->q35.S_slot[l][slot], 0,
                        (size_t)NVH*SD*SD*sizeof(__nv_bfloat16), st);
        cudaMemsetAsync(eng->q35.ring_slot[l][slot], 0,
                        (size_t)CONVD*(CK-1)*sizeof(float), st);
    }
}

// ── Qwen3.5 (qwen35) BATCHED single-pass prefill — kills the token-by-token 149 s ────────────
// P1: prefill ONE sequence's `n` tokens at absolute positions [base, base+n) in CHUNKS of up to
// GEMMA4_MAX_SEQS rows, ONE weight pass per chunk (the GEMV/GEMM projections + FFN are batched
// over the chunk's rows) instead of one full weight pass PER TOKEN. This is the only thing that
// made qwen35 prefill 149 s for a ~1200-token prompt: token-by-token re-streamed all 5 GiB of
// weights once per token. Here every weight read amortizes over the chunk.
//
// Per layer the chunk forward is the proven M4 batched body, except the two GDN kernels that
// carry a genuine intra-sequence recurrence:
//   • FULL  layers — fully batched: each chunk row writes its K/V to the per-slot FULL cache,
//     then attends [0,pos] (the ragged causal order — row t reads rows [base..base+t] written
//     THIS pass plus everything from earlier chunks). Bit-identical to token-by-token.
//   • LINEAR(GDN) layers — the weight-heavy projections (in_qkv / in_z / in_a / in_b / out and
//     the gated-norm) are BATCHED over the chunk; the two recurrent kernels (causal conv1d ring
//     + delta-rule state S) run token-SEQUENTIALLY within the chunk via the SAME M4 stateful
//     kernels (B=1, row-offset pointers), so the per-slot conv ring + GDN state S advance exactly
//     as token-by-token. The conv/GDN kernels touch no weights, so serializing them is cheap.
//
// Every kernel is the same M3/M4 kernel the 8/8 oracle (qwen35_forward_greedy) uses, so the
// produced per-slot state AND the greedily sampled first token are bit-identical to the oracle.
// State (S / ring / FULL K-V) persists in the per-slot arenas across chunks AND across calls, so
// this is also the chunked-prefill primitive (caller resets the slot once before the first
// chunk). do_sample=1 argmaxes the LAST row of the LAST chunk into *first_tok_out. Returns 0/-1.
// use_decode_gdn: when nonzero, the GDN/conv recurrence runs token-SEQUENTIALLY with the single-
// token DECODE kernels (row-offset pointers) instead of the chunked-scan kernels, so the produced
// per-slot state is BYTE-IDENTICAL to sequential decode (the chunk scan is only ~equal and drifts on
// some sequences). Weight-heavy projections stay batched over T rows (one pass). Used by the lossless
// fast-commit; the verify path keeps use_decode_gdn=0 (the chunk scan is fine for scoring argmax).
static void qwen35_prefill_chunk_body_ex(gemma4_engine_t *eng, int slot, int base, int T,
                                         int want_logits, int use_decode_gdn, cudaStream_t st) {
    const gemma4_model_config_t *c = &eng->cfg;
    const int H=c->hidden_size, HD=M2_HEAD, NQ=c->n_heads, NKV=c->n_kv_global;
    const int INNER=c->ssm_inner_size, CONVD=(2*M2_KEYD+c->ssm_inner_size), NKH=M2_NKH, NVH=(c->ssm_inner_size/M2_SD), SD=M2_SD, TSR=c->ssm_time_step_rank, ROT=M2_ROT;
    const int KEYD=M2_KEYD, VALD=c->ssm_inner_size;
    const int I = c->intermediate, VOC = c->vocab_size, L = c->n_layers;
    const int maxctx = eng->q35.maxctx;
    const float eps = 1e-6f;
    const size_t smGDNchunk = ((size_t)SD * SD + M2_CHUNK * SD + 3 * M2_CHUNK) * sizeof(float);
    const size_t smGDN = ((size_t)SD * SD + 3 * SD) * sizeof(float);   // single-token GDN (fast-commit)
    const size_t smATT = (size_t)maxctx * sizeof(float);
    auto Wq = [&](uint64_t off) -> const uint8_t* { return weight_fp8(eng, off); };
    auto Wf = [&](uint64_t off) -> const float*   {
        return (const float*)(eng->d_weights + (off - eng->tdata_start)); };
    int   *d_pos  = workspace_data<int>(eng->q35.prefill_position_workspace);
    int   *d_slot = workspace_data<int>(eng->q35.routing_workspace);
    int32_t *d_tok = workspace_data<int32_t>(eng->q35.prefill_token_workspace);
    auto ws=[&](int id){ return workspace_data<float>(eng->q35.decode_workspace[id]); };
    float *x=ws(Q35_X), *xn=ws(Q35_XN), *qg=ws(Q35_QG),
          *qb=ws(Q35_QB), *gate=ws(Q35_GATE), *kb=ws(Q35_KB),
          *vb=ws(Q35_VB), *attn=ws(Q35_ATTN), *mix=ws(Q35_MIX),
          *qkv=ws(Q35_QKV), *conv=ws(Q35_CONV), *zc=ws(Q35_ZC),
          *ac=ws(Q35_AC), *bc=ws(Q35_BC), *gg=ws(Q35_GG),
          *bb=ws(Q35_BB), *qh=ws(Q35_QH), *kh=ws(Q35_KH),
          *vh=ws(Q35_VH), *core=ws(Q35_CORE), *gnorm=ws(Q35_GNORM),
          *fg=ws(Q35_FG), *fu=ws(Q35_FU), *fa=ws(Q35_FA);
    auto grid1d = [](size_t n){ return (unsigned)((n + 255) / 256); };
    const int NPAD = ((T + M2_CHUNK - 1) / M2_CHUNK) * M2_CHUNK;   // GDN chunk-scan padded length

    embed_w(eng, x, eng->d_token_embd, d_tok, T, H, st);

    for (int l = 0; l < L; l++) {
        const auto &Tn = eng->tensors.layers[l];
        rms_norm_rows_kernel<<<T,256,32*sizeof(float),st>>>(xn, x, Wf(Tn.attn_norm), H, T, eps);
        moe_profile_activation(eng, l, 0, xn, (size_t)T*H, st);
        if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
            q35_proj_gemm(eng, qg, Tn.attn_q, Tn.fmt_q, xn, H, 2*NQ*HD, T, st, l, WC_Q, &Tn.ref_q);
            q35_proj_gemm(eng, kb, Tn.attn_k, Tn.fmt_k, xn, H, NKV*HD,  T, st, l, WC_K, &Tn.ref_k);
            q35_proj_gemm(eng, vb, Tn.attn_v, Tn.fmt_v, xn, H, NKV*HD,  T, st, l, WC_V, &Tn.ref_v);
            m2_split_query_gate_kernel<<<grid1d((size_t)T*NQ*HD),256,0,st>>>(qb, gate, qg, T, NQ);
            per_head_rms_norm_rows_kernel<<<dim3(NQ,T),256,32*sizeof(float),st>>>(qb, Wf(Tn.attn_q_norm), NQ, HD, T, eps);
            per_head_rms_norm_rows_kernel<<<dim3(NKV,T),256,32*sizeof(float),st>>>(kb, Wf(Tn.attn_k_norm), NKV, HD, T, eps);
            qwen35_b_rope_kernel<<<dim3((ROT/2+31)/32,NQ,T),32,0,st>>>(qb, NQ, d_pos, T);
            qwen35_b_rope_kernel<<<dim3((ROT/2+31)/32,NKV,T),32,0,st>>>(kb, NKV, d_pos, T);
            qwen35_b_kv_write_kernel<<<dim3(grid1d((size_t)NKV*HD),T),256,0,st>>>(
                eng->q35.Kc[l], eng->q35.Vc[l], kb, vb, d_pos, d_slot, maxctx, T, NKV);
            // base==0: the whole causal prompt is this tile → tensor-core GEMM attention (reads
            // kb/vb directly). base>0 (chunked continuation): tensor-core GEMMs over the fp16
            // K/V cache in place (q35_cont_attn_tc); the scalar kernel remains only as the
            // alloc/cuBLAS-failure fallback (and under the parity-test override).
            extern int g_fucina_q35_scalar_cont_attn;
            if (base == 0 && T <= 8192 && ensure_q35_attn_scratch(eng, T))
                q35_full_attn_tc(eng, attn, qb, kb, vb, T, st);
            else if (base > 0 && !g_fucina_q35_scalar_cont_attn &&
                     q35_cont_attn_tc(eng, attn, qb, l, slot, base, T, st))
                ;   // tensor-core continuation done
            else
                qwen35_b_attn_kernel<<<dim3(NQ,T),256,smATT,st>>>(
                    attn, qb, eng->q35.Kc[l], eng->q35.Vc[l], d_pos, d_slot, maxctx, T, NKV, NQ);
            m2_sigmoid_gate_mul_kernel<<<grid1d((size_t)T*NQ*HD),256,0,st>>>(attn, gate, T*NQ*HD);
            moe_profile_activation(eng, l, 1, attn, (size_t)T*NQ*HD, st);
            q35_proj_gemm(eng, mix, Tn.attn_output, Tn.fmt_o, attn, NQ*HD, H, T, st, l, WC_O, &Tn.ref_o);
        } else {
            // Weight-heavy projections via tensor-core BF16 GEMM (prefill is compute-bound). The
            // tiny alpha/beta projections (out_dim=TSR=32) stay on the dp4a GEMV — GEMM not worth it.
            q35_proj_gemm(eng, qkv, Tn.ssm.in_qkv, Tn.ssm.fmt_in_qkv, xn, H, CONVD, T, st, l, WC_QKV, &Tn.ssm.ref_in_qkv);
            q35_proj_gemm(eng, zc,  Tn.ssm.in_z,   Tn.ssm.fmt_in_z,   xn, H, INNER, T, st, l, WC_Z, &Tn.ssm.ref_in_z);
            if (eng->format == FORMAT_FP8_BLOCK) {   // in_a/in_b are f32 (oracle parity)
                m2_gemm(ac, xn, Wf(Tn.ssm.in_a), T, H, TSR, st);   // alpha
                m2_gemm(bc, xn, Wf(Tn.ssm.in_b), T, H, TSR, st);   // beta
            } else {
                q35_proj_gemm(eng, ac,  Tn.ssm.in_a, Tn.ssm.fmt_in_a, xn, H, TSR, T, st, l, WC_A, &Tn.ssm.ref_in_a);   // alpha
                q35_proj_gemm(eng, bc,  Tn.ssm.in_b, Tn.ssm.fmt_in_b, xn, H, TSR, T, st, l, WC_B, &Tn.ssm.ref_in_b);   // beta
            }
            m2_decay_beta_kernel<<<grid1d((size_t)T*TSR),256,0,st>>>(gg, bb, ac, bc, Wf(Tn.ssm.a_log), Wf(Tn.ssm.dt_bias), T, TSR);
            if (use_decode_gdn) {
                // LOSSLESS recurrence: run conv + delta-rule state per token with the SINGLE-TOKEN
                // decode kernels (row-offset pointers, B=1), advancing the per-slot ring + state S
                // exactly as sequential decode. Projections above stayed batched (one weight pass).
                for (int t = 0; t < T; t++) {
                    qwen35_b_conv_kernel<<<dim3(grid1d((size_t)CONVD),1),256,0,st>>>(
                        conv + (size_t)t*CONVD, qkv + (size_t)t*CONVD, eng->q35.ring[l], Wf(Tn.ssm.conv1d), d_slot, CONVD, 1);
                    qwen35_b_split_qkv_kernel<<<dim3(grid1d((size_t)CONVD),1),256,0,st>>>(
                        qh + (size_t)t*KEYD, kh + (size_t)t*KEYD, vh + (size_t)t*VALD, conv + (size_t)t*CONVD, 1, CONVD, VALD);
                    m2_l2norm_heads_kernel<<<dim3(NKH,1),128,0,st>>>(qh + (size_t)t*KEYD, NKH, SD, 1);
                    m2_l2norm_heads_kernel<<<dim3(NKH,1),128,0,st>>>(kh + (size_t)t*KEYD, NKH, SD, 1);
                    qwen35_b_gdn_kernel<<<dim3(NVH,1),256,smGDN,st>>>(
                        core + (size_t)t*VALD, qh + (size_t)t*KEYD, kh + (size_t)t*KEYD, vh + (size_t)t*VALD,
                        gg + (size_t)t*TSR, bb + (size_t)t*TSR, eng->q35.S[l], d_slot,
                        1, eng->format==FORMAT_FP8_BLOCK, NVH);
                }
            } else {
            // RECURRENT: causal conv1d ring — ONE batched launch over the whole T-row tile,
            // reading the per-slot ring for the CK-1 carry positions; then advance the ring.
            qwen35_b_conv_chunk_kernel<<<dim3(grid1d((size_t)CONVD),T),256,0,st>>>(
                conv, qkv, eng->q35.ring[l], Wf(Tn.ssm.conv1d), slot, CONVD, T);
            qwen35_b_ring_update_kernel<<<grid1d((size_t)CONVD),256,0,st>>>(
                eng->q35.ring[l], qkv, slot, CONVD, T);
            qwen35_b_split_qkv_kernel<<<dim3(grid1d((size_t)CONVD),T),256,0,st>>>(qh, kh, vh, conv, T, CONVD, VALD);
            m2_l2norm_heads_kernel<<<dim3(NKH,T),128,0,st>>>(qh, NKH, SD, T);
            m2_l2norm_heads_kernel<<<dim3(NKH,T),128,0,st>>>(kh, NKH, SD, T);
            // RECURRENT: delta-rule state S update — ONE chunked parallel-scan launch over the
            // whole tile (CHUNK=64), carrying the per-slot fp32 state S in/out of the arena.
            // 512 threads: the O(SD*SD)=16384-element state/matmul loops stride over 4x the lanes.
            qwen35_b_gdn_chunk_kernel<<<NVH,512,smGDNchunk,st>>>(
                core, qh, kh, vh, gg, bb, eng->q35.S[l],
                workspace_data<float>(eng->q35.gdn_workspace),slot,T,NPAD,
                eng->format==FORMAT_FP8_BLOCK, NVH);
            }
            m2_gated_norm_kernel<<<dim3(NVH,T),128,0,st>>>(gnorm, core, zc, Wf(Tn.ssm.norm), T, NVH, VALD);
            moe_profile_activation(eng, l, 1, gnorm, (size_t)T*INNER, st);
            q35_proj_gemm(eng, mix, Tn.ssm.out, Tn.ssm.fmt_out, gnorm, INNER, H, T, st, l, WC_OUT, &Tn.ssm.ref_out);
        }
        residual_add_kernel<<<grid1d((size_t)T*H),256,0,st>>>(x, mix, T*H);
        rms_norm_rows_kernel<<<T,256,32*sizeof(float),st>>>(xn, x, Wf(Tn.ffn_norm), H, T, eps);
        moe_profile_activation(eng, l, 2, xn, (size_t)T*H, st);
        if (c->n_experts > 0) {   // Qwen3.5-MoE sparse block (experts + shared) → mix
            moe_ffn(eng, l, xn, mix, T, st);
        } else {
            q35_proj_gemm(eng, fg, Tn.ffn_gate, Tn.fmt_gate, xn, H, I, T, st, l, WC_GATE, &Tn.ref_gate);
            q35_proj_gemm(eng, fu, Tn.ffn_up,   Tn.fmt_up,   xn, H, I, T, st, l, WC_UP, &Tn.ref_up);
            silu_glu_kernel<<<grid1d((size_t)T*I),256,0,st>>>(fa, fg, fu, T*I);
            q35_proj_gemm(eng, mix, Tn.ffn_down, Tn.fmt_down, fa, I, H, T, st, l, WC_DOWN, &Tn.ref_down);
        }
        residual_add_kernel<<<grid1d((size_t)T*H),256,0,st>>>(x, mix, T*H);
        q35_jspace_after_layer(eng, x, T, l, st);
        // S1a P4: DFlash aux-hidden capture in the CHUNK body (the verify_block path runs here with
        // want_logits=2). Same gated, additive hook as the decode body: capture ALL T rows' residual
        // at configured target layers into dflash_aux [feature_slot][row][H]. No-op when inactive.
        if (eng->q35.dflash_capture_active && eng->q35.dflash_capture_layer[l]) {
            int fslot = eng->q35.dflash_capture_slot[l];
            int maxr = eng->q35.dflash_capture_maxrows;
            int rr = (T < maxr) ? T : maxr;
            cudaMemcpyAsync(eng->q35.dflash_aux + (size_t)fslot * maxr * H,
                            x, (size_t)rr * H * sizeof(float), cudaMemcpyDeviceToDevice, st);
        }
    }

    if (want_logits == 1) {
        // Final norm over all T rows, but the LM head (VOC=248320, the most expensive GEMV) only on
        // the LAST row — the only one whose logits the caller samples for the first generated token.
        rms_norm_rows_kernel<<<T,256,32*sizeof(float),st>>>(xn, x, Wf(eng->tensors.output_norm), H, T, eps);
        if (eng->format == FORMAT_FP8_BLOCK)   // BF16 untied head (no d_weights lm_head for FP8)
            bf16_head_gemv_launch(eng->d_logits, eng->d_lmhead_bf16, xn + (size_t)(T-1)*H, H, VOC, st);
        else
            gemv_w(eng, eng->d_logits, Wq(eng->tensors.output_weight),
                   xn + (size_t)(T-1)*H, H, VOC, st, eng->tensors.output_fmt);
        argmax_rows_kernel<<<1,1024,0,st>>>(eng->d_logits, eng->d_ms_outtok, 1, VOC);
    } else if (want_logits == 2) {
        // S1a DFlash verify: capture the greedy argmax of EVERY row (the (1+K) verify block). Final
        // norm over all T rows, batched LM head over all rows into d_sb[11], argmax per row into
        // d_ms_outtok[0..T). Same kernels/order as multiseq decode, so per-row argmax equals
        // sequential single-token decode (the losslessness contract the verify path relies on).
        rms_norm_rows_kernel<<<T,256,32*sizeof(float),st>>>(xn, x, Wf(eng->tensors.output_norm), H, T, eps);
        if (eng->format == FORMAT_FP8_BLOCK) {
            float *xt = workspace_data<float>(eng->q35.decode_workspace[Q35_QG]);
            for (int r0 = 0; r0 < T; r0 += 32) {
                int KK = (T - r0 < 32) ? (T - r0) : 32;
                nvfp4_xT_launch(xt, xn + (size_t)r0 * H, H, KK, st);
                bf16_head_gemv_batched_launch(eng->d_sb[11] + (size_t)r0 * VOC,
                                              eng->d_lmhead_bf16, xt, H, VOC, KK, st);
            }
        } else
            gemv_batched_w(eng, eng->d_sb[11], Wq(eng->tensors.output_weight), xn, H, VOC, T, st, eng->tensors.output_fmt);
        argmax_rows_kernel<<<T,1024,0,st>>>(eng->d_sb[11], eng->d_ms_outtok, T, VOC);
    }
}

// Back-compat wrapper: verify/prefill path uses the chunked GDN scan (use_decode_gdn=0).
static void qwen35_prefill_chunk_body(gemma4_engine_t *eng, int slot, int base, int T,
                                      int want_logits, cudaStream_t st) {
    qwen35_prefill_chunk_body_ex(eng, slot, base, T, want_logits, /*use_decode_gdn=*/0, st);
}

static int qwen35_prefill_batched(gemma4_engine_t *eng, int slot, const int32_t *tokens, int n,
                                  int base, int do_sample, int32_t *first_tok_out) {
    if (getenv("FUCINA_P0_DIAG")) {
        fprintf(stderr, "P0DIAG enter slot=%d tok0=%d n=%d base=%d\n", slot, tokens?tokens[0]:-1, n, base);
        fflush(stderr);
    }
    if (!eng || !eng->loaded || slot < 0 || slot >= eng->q35.capacity || !tokens || n <= 0) return -1;
    if (ensure_spec_scratch(eng) != 0 || ensure_q35_scratch(eng) != 0) return -1;
    if (base + n > eng->q35.maxctx) return -1;
    cudaStream_t st = eng->stream;
    const int CH = QWEN35_PF_TILE;             // wide prefill tile (amortizes the 5 GB weight read)
    int h_slot[QWEN35_PF_TILE], h_pos[QWEN35_PF_TILE];
    const bool timing=eng->q35.prefill_timing && n > 2 * FP8_MAXB;
    cudaEvent_t timing_start=nullptr, timing_stop=nullptr;
    if (timing) {
        eng->q35.prefill_dequant_ms=eng->q35.prefill_router_ms=0;
        eng->q35.prefill_expert_ms=eng->q35.prefill_shared_ms=0;
        cudaEventCreate(&timing_start); cudaEventCreate(&timing_stop);
        cudaEventRecord(timing_start,st);
    }
    int done = 0;
    while (done < n) {
        int T = n - done; if (T > CH) T = CH;
        int b = base + done;
        for (int r = 0; r < T; r++) { h_slot[r] = slot; h_pos[r] = b + r; }
        cudaMemcpyAsync(workspace_data<int32_t>(eng->q35.prefill_token_workspace),tokens+done,
                        (size_t)T*sizeof(int32_t),cudaMemcpyHostToDevice,st);
        cudaMemcpyAsync(workspace_data<int>(eng->q35.prefill_position_workspace),h_pos,
                        (size_t)T*sizeof(int),cudaMemcpyHostToDevice,st);
        cudaMemcpyAsync(workspace_data<int>(eng->q35.routing_workspace),h_slot,
                        (size_t)T*sizeof(int),cudaMemcpyHostToDevice,st);
        int want = (do_sample && done + T >= n);   // sample only the final row of the final chunk
        qwen35_prefill_chunk_body(eng, slot, b, T, want, st);
        done += T;
    }
    int32_t first = 0;
    if (do_sample) {
        gemma4_seq *one[1] = { &eng->slots[slot] };
        if (eng->slots[slot].samp_temp > 0.f)
            qwen35_sample_rows(eng, one, eng->d_logits, 1, st);
        cudaMemcpyAsync(&first, eng->d_ms_outtok, sizeof(int32_t), cudaMemcpyDeviceToHost, st);
    }
    float total_ms=0;
    if (timing) {
        cudaEventRecord(timing_stop,st); cudaEventSynchronize(timing_stop);
        cudaEventElapsedTime(&total_ms,timing_start,timing_stop);
        cudaEventDestroy(timing_start); cudaEventDestroy(timing_stop);
        double measured=eng->q35.prefill_dequant_ms+eng->q35.prefill_router_ms+
                        eng->q35.prefill_expert_ms+eng->q35.prefill_shared_ms;
        double other=(double)total_ms-measured; if(other<0) other=0;
        fprintf(stderr,
                "fucina: qwen35 prefill phases: tokens=%d base=%d tiles=%d total=%.2f ms "
                "expert-dequant=%.2f router-route=%.2f grouped-experts=%.2f "
                "shared-expert=%.2f other=%.2f sum=%.1f%%\n",
                n,base,(n+CH-1)/CH,total_ms,eng->q35.prefill_dequant_ms,
                eng->q35.prefill_router_ms,eng->q35.prefill_expert_ms,
                eng->q35.prefill_shared_ms,other,total_ms>0?100.0*(measured+other)/total_ms:0.0);
    } else cudaStreamSynchronize(st);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (do_sample && first_tok_out) *first_tok_out = first;
    if (getenv("FUCINA_P0_DIAG")) {
        fprintf(stderr, "P0DIAG exit  slot=%d tok0=%d n=%d base=%d first=%d\n", slot, tokens?tokens[0]:-1, n, base, first);
        fflush(stderr);
    }
    return 0;
}

// ── Qwen3.5 BATCHED MULTI-SEQUENCE prefill (P1: kills serial admission-prefill) ──────────────
// Prefill M freshly-opened sequences in ONE forward. Each seq i occupies rows [offs[i],offs[i]+
// lens[i]) of the flattened Ttot-row buffers at absolute positions [0,lens[i]) (fresh slots).
// The weight-heavy work (projections / MoE experts / router / norms / FULL-attn K/V writes) is
// BATCHED over all Ttot rows — one weight read amortized across every sequence, instead of M
// separate single-seq forwards each re-streaming the whole MoE weight set (~52 ms each measured).
// The genuinely per-sequence pieces stay per-sequence: FULL-attn uses the scalar per-row path
// (each row attends its OWN slot's KV via d_slot/d_pos — already multi-seq), and the GDN/conv
// recurrence runs once PER SEQUENCE with row-offset pointers keyed to that seq's slot (the delta
// state S and conv ring are per-slot recurrences that must NOT be batched across sequences).
// LOSSLESS: every per-row op is row-independent (batching a bigger GEMM does not change a row's
// value) and every per-seq recurrence is the SAME kernel a standalone prefill runs → first token
// + continuation byte-identical to per-sequence qwen35_prefill_batched. Caller must have reset
// each slot (n_tokens==0, S/ring zeroed) via the seq_add prologue. d_tok/d_pos/d_slot must be
// uploaded for the Ttot rows before this call. Argmaxes each seq's LAST row into d_ms_outtok[i].
static void qwen35_prefill_multiseq_body(gemma4_engine_t *eng, const int *slots,
                                         const int *offs, const int *lens, int M, int Ttot,
                                         cudaStream_t st) {
    const gemma4_model_config_t *c = &eng->cfg;
    const int H=c->hidden_size, HD=M2_HEAD, NQ=c->n_heads, NKV=c->n_kv_global;
    const int INNER=c->ssm_inner_size, CONVD=(2*M2_KEYD+c->ssm_inner_size), NKH=M2_NKH, NVH=(c->ssm_inner_size/M2_SD), SD=M2_SD, TSR=c->ssm_time_step_rank, ROT=M2_ROT;
    const int VALD=c->ssm_inner_size;
    const int I = c->intermediate, VOC = c->vocab_size, L = c->n_layers;
    const int maxctx = eng->q35.maxctx;
    const float eps = 1e-6f;
    const size_t smGDNchunk = ((size_t)SD * SD + M2_CHUNK * SD + 3 * M2_CHUNK) * sizeof(float);
    const size_t smATT = (size_t)maxctx * sizeof(float);
    auto Wf = [&](uint64_t off) -> const float* {
        return (const float*)(eng->d_weights + (off - eng->tdata_start)); };
    int   *d_pos  = workspace_data<int>(eng->q35.prefill_position_workspace);
    int   *d_slot = workspace_data<int>(eng->q35.routing_workspace);
    int32_t *d_tok = workspace_data<int32_t>(eng->q35.prefill_token_workspace);
    auto ws=[&](int id){ return workspace_data<float>(eng->q35.decode_workspace[id]); };
    float *x=ws(Q35_X), *xn=ws(Q35_XN), *qg=ws(Q35_QG),
          *qb=ws(Q35_QB), *gate=ws(Q35_GATE), *kb=ws(Q35_KB),
          *vb=ws(Q35_VB), *attn=ws(Q35_ATTN), *mix=ws(Q35_MIX),
          *qkv=ws(Q35_QKV), *conv=ws(Q35_CONV), *zc=ws(Q35_ZC),
          *ac=ws(Q35_AC), *bc=ws(Q35_BC), *gg=ws(Q35_GG),
          *bb=ws(Q35_BB), *qh=ws(Q35_QH), *kh=ws(Q35_KH),
          *vh=ws(Q35_VH), *core=ws(Q35_CORE), *gnorm=ws(Q35_GNORM),
          *fg=ws(Q35_FG), *fu=ws(Q35_FU), *fa=ws(Q35_FA);
    auto grid1d = [](size_t n){ return (unsigned)((n + 255) / 256); };
    const int T = Ttot;

    embed_w(eng, x, eng->d_token_embd, d_tok, T, H, st);

    for (int l = 0; l < L; l++) {
        const auto &Tn = eng->tensors.layers[l];
        rms_norm_rows_kernel<<<T,256,32*sizeof(float),st>>>(xn, x, Wf(Tn.attn_norm), H, T, eps);
        moe_profile_activation(eng, l, 0, xn, (size_t)T*H, st);
        if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
            q35_proj_gemm(eng, qg, Tn.attn_q, Tn.fmt_q, xn, H, 2*NQ*HD, T, st, l, WC_Q, &Tn.ref_q);
            q35_proj_gemm(eng, kb, Tn.attn_k, Tn.fmt_k, xn, H, NKV*HD,  T, st, l, WC_K, &Tn.ref_k);
            q35_proj_gemm(eng, vb, Tn.attn_v, Tn.fmt_v, xn, H, NKV*HD,  T, st, l, WC_V, &Tn.ref_v);
            m2_split_query_gate_kernel<<<grid1d((size_t)T*NQ*HD),256,0,st>>>(qb, gate, qg, T, NQ);
            per_head_rms_norm_rows_kernel<<<dim3(NQ,T),256,32*sizeof(float),st>>>(qb, Wf(Tn.attn_q_norm), NQ, HD, T, eps);
            per_head_rms_norm_rows_kernel<<<dim3(NKV,T),256,32*sizeof(float),st>>>(kb, Wf(Tn.attn_k_norm), NKV, HD, T, eps);
            qwen35_b_rope_kernel<<<dim3((ROT/2+31)/32,NQ,T),32,0,st>>>(qb, NQ, d_pos, T);
            qwen35_b_rope_kernel<<<dim3((ROT/2+31)/32,NKV,T),32,0,st>>>(kb, NKV, d_pos, T);
            qwen35_b_kv_write_kernel<<<dim3(grid1d((size_t)NKV*HD),T),256,0,st>>>(
                eng->q35.Kc[l], eng->q35.Vc[l], kb, vb, d_pos, d_slot, maxctx, T, NKV);
            // MULTI-SEQ FULL-attn: per SEQUENCE, via the SAME base==0 tensor-core kernel the
            // standalone prefill uses (byte-identical: same kernel, same rows, same length) —
            // NOT one batched call (that would mix rows across sequences). Scalar cache-read path
            // is the per-seq memory-pressure fallback (d_pos/d_slot offset to that seq's rows).
            for (int i = 0; i < M; i++) {
                int aoff = offs[i], alen = lens[i];
                if (alen <= 8192 && ensure_q35_attn_scratch(eng, alen))
                    q35_full_attn_tc(eng, attn + (size_t)aoff*NQ*HD, qb + (size_t)aoff*NQ*HD,
                                     kb + (size_t)aoff*NKV*HD, vb + (size_t)aoff*NKV*HD, alen, st);
                else
                    qwen35_b_attn_kernel<<<dim3(NQ,alen),256,smATT,st>>>(
                        attn + (size_t)aoff*NQ*HD, qb + (size_t)aoff*NQ*HD, eng->q35.Kc[l], eng->q35.Vc[l],
                        d_pos + aoff, d_slot + aoff, maxctx, alen, NKV, NQ);
            }
            m2_sigmoid_gate_mul_kernel<<<grid1d((size_t)T*NQ*HD),256,0,st>>>(attn, gate, T*NQ*HD);
            moe_profile_activation(eng, l, 1, attn, (size_t)T*NQ*HD, st);
            q35_proj_gemm(eng, mix, Tn.attn_output, Tn.fmt_o, attn, NQ*HD, H, T, st, l, WC_O, &Tn.ref_o);
        } else {
            q35_proj_gemm(eng, qkv, Tn.ssm.in_qkv, Tn.ssm.fmt_in_qkv, xn, H, CONVD, T, st, l, WC_QKV, &Tn.ssm.ref_in_qkv);
            q35_proj_gemm(eng, zc,  Tn.ssm.in_z,   Tn.ssm.fmt_in_z,   xn, H, INNER, T, st, l, WC_Z, &Tn.ssm.ref_in_z);
            if (eng->format == FORMAT_FP8_BLOCK) {
                m2_gemm(ac, xn, Wf(Tn.ssm.in_a), T, H, TSR, st);
                m2_gemm(bc, xn, Wf(Tn.ssm.in_b), T, H, TSR, st);
            } else {
                q35_proj_gemm(eng, ac,  Tn.ssm.in_a, Tn.ssm.fmt_in_a, xn, H, TSR, T, st, l, WC_A, &Tn.ssm.ref_in_a);
                q35_proj_gemm(eng, bc,  Tn.ssm.in_b, Tn.ssm.fmt_in_b, xn, H, TSR, T, st, l, WC_B, &Tn.ssm.ref_in_b);
            }
            // RECURRENT conv1d ring — PER SEQUENCE (each keyed to its own slot ring), row-offset
            // pointers into the flattened Ttot-row qkv/conv buffers.
            for (int i = 0; i < M; i++) {
                int off = offs[i], len = lens[i], slot = slots[i];
                qwen35_b_conv_chunk_kernel<<<dim3(grid1d((size_t)CONVD),len),256,0,st>>>(
                    conv + (size_t)off*CONVD, qkv + (size_t)off*CONVD, eng->q35.ring[l], Wf(Tn.ssm.conv1d), slot, CONVD, len);
                qwen35_b_ring_update_kernel<<<grid1d((size_t)CONVD),256,0,st>>>(
                    eng->q35.ring[l], qkv + (size_t)off*CONVD, slot, CONVD, len);
            }
            qwen35_b_split_qkv_kernel<<<dim3(grid1d((size_t)CONVD),T),256,0,st>>>(qh, kh, vh, conv, T, CONVD, VALD);
            m2_l2norm_heads_kernel<<<dim3(NKH,T),128,0,st>>>(qh, NKH, SD, T);
            m2_l2norm_heads_kernel<<<dim3(NKH,T),128,0,st>>>(kh, NKH, SD, T);
            m2_decay_beta_kernel<<<grid1d((size_t)T*TSR),256,0,st>>>(gg, bb, ac, bc, Wf(Tn.ssm.a_log), Wf(Tn.ssm.dt_bias), T, TSR);
            // RECURRENT delta-rule state S — PER SEQUENCE (each seq's tokens are one contiguous
            // 64-token chunked scan carrying S[l][slot_i]), row-offset pointers.
            for (int i = 0; i < M; i++) {
                int off = offs[i], len = lens[i], slot = slots[i];
                int NPAD_i = ((len + M2_CHUNK - 1) / M2_CHUNK) * M2_CHUNK;
                qwen35_b_gdn_chunk_kernel<<<NVH,512,smGDNchunk,st>>>(
                    core + (size_t)off*NVH*SD, qh + (size_t)off*NKH*SD, kh + (size_t)off*NKH*SD,
                    vh + (size_t)off*NVH*SD, gg + (size_t)off*NVH, bb + (size_t)off*NVH, eng->q35.S[l],
                    workspace_data<float>(eng->q35.gdn_workspace), slot, len, NPAD_i,
                    eng->format==FORMAT_FP8_BLOCK, NVH);
            }
            m2_gated_norm_kernel<<<dim3(NVH,T),128,0,st>>>(gnorm, core, zc, Wf(Tn.ssm.norm), T, NVH, VALD);
            moe_profile_activation(eng, l, 1, gnorm, (size_t)T*INNER, st);
            q35_proj_gemm(eng, mix, Tn.ssm.out, Tn.ssm.fmt_out, gnorm, INNER, H, T, st, l, WC_OUT, &Tn.ssm.ref_out);
        }
        residual_add_kernel<<<grid1d((size_t)T*H),256,0,st>>>(x, mix, T*H);
        rms_norm_rows_kernel<<<T,256,32*sizeof(float),st>>>(xn, x, Wf(Tn.ffn_norm), H, T, eps);
        moe_profile_activation(eng, l, 2, xn, (size_t)T*H, st);
        if (c->n_experts > 0) {
            moe_ffn(eng, l, xn, mix, T, st);
        } else {
            q35_proj_gemm(eng, fg, Tn.ffn_gate, Tn.fmt_gate, xn, H, I, T, st, l, WC_GATE, &Tn.ref_gate);
            q35_proj_gemm(eng, fu, Tn.ffn_up,   Tn.fmt_up,   xn, H, I, T, st, l, WC_UP, &Tn.ref_up);
            silu_glu_kernel<<<grid1d((size_t)T*I),256,0,st>>>(fa, fg, fu, T*I);
            q35_proj_gemm(eng, mix, Tn.ffn_down, Tn.fmt_down, fa, I, H, T, st, l, WC_DOWN, &Tn.ref_down);
        }
        residual_add_kernel<<<grid1d((size_t)T*H),256,0,st>>>(x, mix, T*H);
        q35_jspace_after_layer(eng, x, T, l, st);
    }

    // Final norm over all rows, then the batched LM head on ONLY the M last-rows (one per seq).
    rms_norm_rows_kernel<<<T,256,32*sizeof(float),st>>>(xn, x, Wf(eng->tensors.output_norm), H, T, eps);
    // Gather each sequence's last row into a contiguous M×H buffer (attn is free at head time).
    float *xhead = attn;
    for (int i = 0; i < M; i++)
        cudaMemcpyAsync(xhead + (size_t)i*H, xn + (size_t)(offs[i]+lens[i]-1)*H,
                        (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, st);
    // BF16 head per row — the SAME path gemma4_engine_seq_add uses (bf16_head_gemv_launch), so the
    // batched first token is bit-identical to standalone (the FP4-activation decode head would flip
    // tiny-margin argmaxes and is a first-token accuracy regression vs seq_add). One GEMV per row.
    if (eng->format == FORMAT_FP8_BLOCK) {
        for (int i = 0; i < M; i++)
            bf16_head_gemv_launch(eng->d_sb[11] + (size_t)i * VOC, eng->d_lmhead_bf16,
                                  xhead + (size_t)i * H, H, VOC, st);
    } else {
        for (int i = 0; i < M; i++)
            gemv_w(eng, eng->d_sb[11] + (size_t)i * VOC, weight_fp8(eng, eng->tensors.output_weight),
                   xhead + (size_t)i * H, H, VOC, st, eng->tensors.output_fmt);
    }
    argmax_rows_kernel<<<M,1024,0,st>>>(eng->d_sb[11], eng->d_ms_outtok, M, VOC);
}

static int qwen35_seq_add(gemma4_engine_t *eng, const int32_t *prompt, int n_prompt,
                          int32_t *first_token_out, float temp, int top_k, float top_p,
                          float min_p, uint64_t seed) {
    if (!eng || !eng->loaded || !prompt || n_prompt <= 0) return -1;
    if (ensure_spec_scratch(eng) != 0 || ensure_q35_scratch(eng) != 0) return -1;
    if (n_prompt > eng->q35.maxctx) return -1;
    int slot = -1;
    for (int i = 0; i < eng->q35.capacity; i++) if (!eng->slots[i].used) { slot = i; break; }
    if (slot < 0 || q35_slot_state_ensure(eng, slot) != 0 ||
        q35_slot_kv_reserve(eng, slot, n_prompt) != 0) return -1;
    gemma4_seq *s = &eng->slots[slot];
    s->used = 1; s->n_tokens = 0; s->samp_temp = temp; s->samp_top_k = top_k;
    s->samp_top_p = top_p; s->samp_min_p = min_p; s->samp_seed = seed;
    s->n_sampled = 0; s->mtp_h_valid = 0;
    cudaStream_t st = eng->stream;
    qwen35_slot_reset(eng, slot, st);
    // P1: BATCHED single-pass prefill (one weight pass per ≤GEMMA4_MAX_SEQS-token chunk) instead of
    // token-by-token (which re-streamed all weights per token → 149 s). g_fucina_force_slow_prefill
    // forces the proven token-by-token loop for the dual-path determinism self-test.
    extern int g_fucina_force_slow_prefill;
    if (!g_fucina_force_slow_prefill) {
        int32_t ft = 0;
        if (qwen35_prefill_batched(eng, slot, prompt, n_prompt, /*base=*/0, /*do_sample=*/1, &ft) != 0) {
            s->used = 0; return -1;
        }
        s->n_tokens = n_prompt; s->n_sampled = 1;
        if (first_token_out) *first_token_out = ft;
        return slot;
    }
    gemma4_seq *one[1] = { s };
    for (int i = 0; i < n_prompt; i++) {   // sequential prefill (GDN recurrence is per-position)
        int pos = i; int32_t tok = prompt[i]; int want = (i == n_prompt - 1);
        qwen35_ms_run(eng, one, &tok, &pos, 1, want, /*use_graph=*/0, /*splice=*/0);
    }
    int32_t first = 0;
    cudaMemcpyAsync(&first, eng->d_ms_outtok, sizeof(int32_t), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    { cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) { fprintf(stderr, "qwen35_seq_add: CUDA error during prefill: %s\n",
                                      cudaGetErrorString(e)); s->used = 0; return -1; } }
    s->n_tokens = n_prompt; s->n_sampled = 1;
    if (first_token_out) *first_token_out = first;
    return slot;
}

// P1: admit M sequences in ONE batched prefill forward (the anti-serialization lever). Allocates
// M fresh slots, resets each, lays out the concatenated prompts as Ttot rows, runs the batched
// multi-seq prefill body (weights amortized across all rows), samples each seq's first token.
// Byte-identical greedy to M separate qwen35_seq_add calls. Returns M on success (out_slots/
// out_first filled), -1 on any error (all allocated slots freed). Ttot must fit the prefill tile.
static int qwen35_seq_add_multiseq(gemma4_engine_t *eng, const int32_t *tokens_flat,
                                   const int *lens, int M, const float *temps, const int *topks,
                                   const float *topps, const float *minps, const uint64_t *seeds,
                                   int *out_slots, int32_t *out_first) {
    if (!eng || !eng->loaded || !tokens_flat || !lens || M <= 0 || M > eng->q35.capacity) return -1;
    if (ensure_spec_scratch(eng) != 0 || ensure_q35_scratch(eng) != 0) return -1;
    int Ttot = 0;
    for (int i = 0; i < M; i++) { if (lens[i] <= 0) return -1; Ttot += lens[i]; }
    if (Ttot > QWEN35_PF_TILE || Ttot > eng->q35.maxctx) return -1;   // caller splits / falls back
    cudaStream_t st = eng->stream;

    int slots[GEMMA4_CAP_EXPERTS], offs[GEMMA4_CAP_EXPERTS];   // M <= capacity (small)
    int nalloc = 0;
    auto rollback = [&](){ for (int j = 0; j < nalloc; j++) eng->slots[slots[j]].used = 0; };
    int off = 0;
    for (int i = 0; i < M; i++) {
        int slot = -1;
        for (int k = 0; k < eng->q35.capacity; k++) if (!eng->slots[k].used) { slot = k; break; }
        if (slot < 0 || q35_slot_state_ensure(eng, slot) != 0 ||
            q35_slot_kv_reserve(eng, slot, lens[i]) != 0) { rollback(); return -1; }
        gemma4_seq *s = &eng->slots[slot];
        s->used = 1; s->n_tokens = 0; s->samp_temp = temps ? temps[i] : 0.f;
        s->samp_top_k = topks ? topks[i] : 0; s->samp_top_p = topps ? topps[i] : 0.f;
        s->samp_min_p = minps ? minps[i] : 0.f; s->samp_seed = seeds ? seeds[i] : 0;
        s->n_sampled = 0; s->mtp_h_valid = 0;
        qwen35_slot_reset(eng, slot, st);
        slots[i] = slot; offs[i] = off; off += lens[i]; nalloc++;
    }

    // Lay out the Ttot flattened rows: token, per-seq position [0,len), per-row slot.
    static int32_t h_tok[QWEN35_PF_TILE]; static int h_pos[QWEN35_PF_TILE], h_slot[QWEN35_PF_TILE];
    for (int i = 0; i < M; i++)
        for (int r = 0; r < lens[i]; r++) {
            int row = offs[i] + r; h_tok[row] = tokens_flat[row]; h_pos[row] = r; h_slot[row] = slots[i];
        }
    cudaMemcpyAsync(workspace_data<int32_t>(eng->q35.prefill_token_workspace), h_tok, (size_t)Ttot*sizeof(int32_t), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(workspace_data<int>(eng->q35.prefill_position_workspace), h_pos, (size_t)Ttot*sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(workspace_data<int>(eng->q35.routing_workspace), h_slot, (size_t)Ttot*sizeof(int), cudaMemcpyHostToDevice, st);

    qwen35_prefill_multiseq_body(eng, slots, offs, lens, M, Ttot, st);

    // Greedy argmax already in d_ms_outtok; re-sample rows with temp>0 (row-independent, matches
    // per-seq qwen35_seq_add which only sample_rows when that seq's temp>0).
    bool any_temp = false;
    for (int i = 0; i < M; i++) if ((temps ? temps[i] : 0.f) > 0.f) { any_temp = true; break; }
    if (any_temp) {
        gemma4_seq *slv[GEMMA4_CAP_EXPERTS];
        for (int i = 0; i < M; i++) slv[i] = &eng->slots[slots[i]];
        qwen35_sample_rows(eng, slv, eng->d_sb[11], M, st);
    }
    int32_t firsts[GEMMA4_CAP_EXPERTS];
    cudaMemcpyAsync(firsts, eng->d_ms_outtok, (size_t)M*sizeof(int32_t), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    if (cudaGetLastError() != cudaSuccess) { rollback(); return -1; }
    for (int i = 0; i < M; i++) {
        gemma4_seq *s = &eng->slots[slots[i]];
        s->n_tokens = lens[i]; s->n_sampled = 1;
        out_slots[i] = slots[i]; out_first[i] = firsts[i];
    }
    return M;
}

static int qwen35_seq_open(gemma4_engine_t *eng, float temp, int top_k, float top_p,
                           float min_p, uint64_t seed) {
    if (!eng || !eng->loaded) return -1;
    if (ensure_spec_scratch(eng) != 0 || ensure_q35_scratch(eng) != 0) return -1;
    int slot = -1;
    for (int i = 0; i < eng->q35.capacity; i++) if (!eng->slots[i].used) { slot = i; break; }
    if (slot < 0 || q35_slot_state_ensure(eng, slot) != 0) return -1;
    gemma4_seq *s = &eng->slots[slot];
    s->used = 1; s->n_tokens = 0; s->samp_temp = temp; s->samp_top_k = top_k;
    s->samp_top_p = top_p; s->samp_min_p = min_p; s->samp_seed = seed;
    s->n_sampled = 0; s->mtp_h_valid = 0;
    qwen35_slot_reset(eng, slot, eng->stream);
    return slot;
}

static int qwen35_seq_prefill_chunk(gemma4_engine_t *eng, int slot, const int32_t *tokens,
                                    int n, int do_sample, int32_t *first_token_out) {
    if (!eng || !eng->loaded || slot < 0 || slot >= eng->q35.capacity || !tokens || n <= 0) return -1;
    if (ensure_q35_scratch(eng) != 0) return -1;
    gemma4_seq *s = &eng->slots[slot];
    if (!s->used) return -1;
    if (s->n_tokens + n > eng->q35.maxctx ||
        q35_slot_kv_reserve(eng, slot, s->n_tokens + n) != 0) return -1;
    cudaStream_t st = eng->stream;
    // P1: BATCHED chunk prefill — one weight pass per ≤GEMMA4_MAX_SEQS-token sub-chunk, appending at
    // the slot's current position (state persists in the per-slot arenas). g_fucina_force_slow_prefill
    // forces the token-by-token loop for the dual-path determinism self-test.
    extern int g_fucina_force_slow_prefill;
    if (!g_fucina_force_slow_prefill) {
        int32_t ft = 0;
        if (qwen35_prefill_batched(eng, slot, tokens, n, /*base=*/s->n_tokens, do_sample, &ft) != 0)
            return -1;
        s->n_tokens += n;
        if (do_sample) s->n_sampled++;
        if (do_sample && first_token_out) *first_token_out = ft;
        return 0;
    }
    gemma4_seq *one[1] = { s };
    for (int i = 0; i < n; i++) {
        int pos = s->n_tokens + i; int32_t tok = tokens[i]; int want = (do_sample && i == n - 1);
        qwen35_ms_run(eng, one, &tok, &pos, 1, want, /*use_graph=*/0, /*splice=*/0);
    }
    int32_t first = 0;
    if (do_sample) cudaMemcpyAsync(&first, eng->d_ms_outtok, sizeof(int32_t), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    if (cudaGetLastError() != cudaSuccess) return -1;
    s->n_tokens += n;
    if (do_sample) s->n_sampled++;
    if (do_sample && first_token_out) *first_token_out = first;
    return 0;
}

// ─── Qwen3.5 hybrid per-slot state snapshot (conversation/state cache) ──────
//
// A hybrid slot's state is (a) a FIXED-size recurrent part per LINEAR layer —
// GDN delta-rule state S [NVH*SD*SD] + causal-conv ring [CONVD*(CK-1)] — and
// (b) a length-proportional FULL-layer K/V prefix [n_tokens*NKV*HD] each (K
// and V, position-major, so the prefix is the leading contiguous slice).
//
// The GDN state is a recurrence at EXACTLY n_tokens: unlike attention KV it
// cannot be truncated to an arbitrary shorter prefix. A snapshot is therefore
// restorable ONLY into a prompt that EXTENDS its token sequence — which is
// precisely the agentic multi-turn case (conversation + new turn appended).
//
// Buffer layout: layers in order; FULL → K slice then V slice; LINEAR → S
// then ring. All fp32 device bytes.

static size_t q35_state_size_bytes(const gemma4_engine_t *eng, int n_tokens) {
    const gemma4_model_config_t *c = &eng->cfg;
    // RUNTIME geometry (27B-dense: NVH=48, CONVD=10240 ≠ the baked M2_* 35B values).
    const int NKV = c->n_kv_global, HD = M2_HEAD;
    const int NVH = c->ssm_time_step_rank, CONVD = 2 * M2_KEYD + c->ssm_inner_size;
    size_t bytes = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (c->attn_kind[l] == GEMMA4_ATTN_FULL)
            bytes += 2ull * (size_t)n_tokens * NKV * HD * sizeof(__half);   // fp16 K + V
        else
            bytes += (size_t)NVH * M2_SD * M2_SD * sizeof(__nv_bfloat16)   // bf16 GDN state
                   + (size_t)CONVD * (M2_CK - 1) * sizeof(float);          // fp32 conv ring
    }
    return bytes;
}

extern "C" size_t gemma4_engine_q35_state_size(gemma4_engine_t *eng, int n_tokens) {
    if (!eng || !eng->loaded || eng->cfg.arch != GEMMA4_ARCH_QWEN3_5 || n_tokens <= 0) return 0;
    if (ensure_q35_scratch(eng) != 0) return 0;
    if (n_tokens > eng->q35.maxctx) return 0;
    return q35_state_size_bytes(eng, n_tokens);
}

// q35_state_copy moves one slot's state (at n_tokens) between the device
// arenas and a host buffer. to_host=1 → save, 0 → restore. Synchronous on the
// engine stream (pageable host memory serializes the copies anyway; on the
// GB10's unified memory this is a plain memcpy at full bandwidth).
static int q35_state_copy(gemma4_engine_t *eng, int slot, char *h, int n_tokens, int to_host) {
    const gemma4_model_config_t *c = &eng->cfg;
    const int NKV = c->n_kv_global, HD = M2_HEAD;
    // RUNTIME geometry: arena strides below must match the ensure_q35_scratch allocations,
    // which use the runtime NVH/CONVD (baked M2_* here silently mis-strided the 27B).
    const int NVH = c->ssm_time_step_rank, CONVD = 2 * M2_KEYD + c->ssm_inner_size;
    const size_t kv = (size_t)n_tokens * NKV * HD * sizeof(__half);   // fp16 K/V slice
    const size_t s_sz = (size_t)NVH * M2_SD * M2_SD * sizeof(__nv_bfloat16);  // bf16 GDN state arena
    const size_t r_sz = (size_t)CONVD * (M2_CK - 1) * sizeof(float);
    cudaStream_t st = eng->stream;
    for (int l = 0; l < c->n_layers; l++) {
        if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
            __half *Kb = eng->q35.Kc_slot[l][slot];
            __half *Vb = eng->q35.Vc_slot[l][slot];
            if (to_host) {
                cudaMemcpyAsync(h, Kb, kv, cudaMemcpyDeviceToHost, st); h += kv;
                cudaMemcpyAsync(h, Vb, kv, cudaMemcpyDeviceToHost, st); h += kv;
            } else {
                cudaMemcpyAsync(Kb, h, kv, cudaMemcpyHostToDevice, st); h += kv;
                cudaMemcpyAsync(Vb, h, kv, cudaMemcpyHostToDevice, st); h += kv;
            }
        } else {
            __nv_bfloat16 *S = eng->q35.S_slot[l][slot];
            float *R = eng->q35.ring_slot[l][slot];
            if (to_host) {
                cudaMemcpyAsync(h, S, s_sz, cudaMemcpyDeviceToHost, st); h += s_sz;
                cudaMemcpyAsync(h, R, r_sz, cudaMemcpyDeviceToHost, st); h += r_sz;
            } else {
                cudaMemcpyAsync(S, h, s_sz, cudaMemcpyHostToDevice, st); h += s_sz;
                cudaMemcpyAsync(R, h, r_sz, cudaMemcpyHostToDevice, st); h += r_sz;
            }
        }
    }
    cudaStreamSynchronize(st);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Pinned host allocation for snapshot buffers — see the .cuh doc.
extern "C" void *gemma4_host_alloc(size_t bytes) {
    void *p = NULL;
    if (cudaMallocHost(&p, bytes) != cudaSuccess) { cudaGetLastError(); return NULL; }
    return p;
}
extern "C" void gemma4_host_free(void *p) {
    if (p) cudaFreeHost(p);
}

extern "C" int gemma4_engine_q35_state_save(gemma4_engine_t *eng, int slot,
                                            void *buf, int n_tokens) {
    if (!eng || !eng->loaded || eng->cfg.arch != GEMMA4_ARCH_QWEN3_5 || !buf) return -1;
    if (slot < 0 || slot >= eng->q35.capacity || !eng->slots[slot].used) return -1;
    if (!eng->q35.ready || eng->slots[slot].n_tokens != n_tokens) return -1;
    return q35_state_copy(eng, slot, (char *)buf, n_tokens, 1);
}

extern "C" int gemma4_engine_q35_state_restore(gemma4_engine_t *eng, int slot,
                                               const void *buf, int n_tokens) {
    if (!eng || !eng->loaded || eng->cfg.arch != GEMMA4_ARCH_QWEN3_5 || !buf) return -1;
    if (slot < 0 || slot >= eng->q35.capacity || !eng->slots[slot].used) return -1;
    if (ensure_q35_scratch(eng) != 0) return -1;
    if (n_tokens <= 0 || n_tokens > eng->q35.maxctx ||
        q35_slot_kv_reserve(eng, slot, n_tokens) != 0) return -1;
    if (q35_state_copy(eng, slot, (char *)buf, n_tokens, 0) != 0) return -1;
    gemma4_seq *s = &eng->slots[slot];
    s->n_tokens = n_tokens;
    s->n_sampled = 0;
    s->mtp_h_valid = 0;
    return 0;
}

// Committed token count of a live slot (-1 if free/invalid). The scheduler's
// per-sequence history can run one token AHEAD of the engine (the last sampled
// token is not fed back until the next step), so snapshot callers use this to
// key the saved state by the tokens actually in the arenas.
extern "C" int gemma4_engine_seq_ntokens(gemma4_engine_t *eng, int slot) {
    if (!eng || slot < 0 || slot >= eng->q35.capacity || !eng->slots[slot].used) return -1;
    return eng->slots[slot].n_tokens;
}

extern "C" void gemma4_engine_memory_stats(const gemma4_engine_t *eng,
                                             gemma4_memory_stats_t *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!eng || eng->cfg.arch != GEMMA4_ARCH_QWEN3_5) return;
    out->qwen_workspace_bytes = eng->q35.workspace_bytes;
    out->qwen_recurrent_per_slot_bytes = eng->q35.per_slot_recurrent_bytes;
    out->qwen_kv_per_slot_bytes = eng->q35.per_slot_kv_bytes;
    out->qwen_committed_bytes = eng->q35.committed_bytes;
    out->qwen_reserved_bytes = eng->q35.reserved_bytes;
    out->qwen_peak_bytes = eng->q35.peak_bytes;
    out->qwen_capacity = eng->q35.capacity;
    out->qwen_allocated_slots = eng->q35.allocated_slots;
    out->qwen_max_context = eng->q35.maxctx;
    out->qwen_reserved_context = eng->q35.reserved_context;
}

// S2b decode-first batch ordering (q35_sort_batch_decode_first) lives in qwen35_graph_key.cuh.
static int qwen35_step_batch(gemma4_engine_t *eng, const int *slots,
                             const int32_t *in_tokens, int B, int32_t *out_tokens) {
    if (!eng || !eng->loaded || B <= 0 || B > eng->q35.capacity) return -1;
    if (ensure_spec_scratch(eng) != 0 || ensure_q35_scratch(eng) != 0) return -1;
    gemma4_seq *slv[GEMMA4_MAX_SEQS]; int positions[GEMMA4_MAX_SEQS];
    int32_t in2[GEMMA4_MAX_SEQS]; int rowmap[GEMMA4_MAX_SEQS]; int Bv = 0;
    for (int r = 0; r < B; r++) {
        int id = slots[r];
        if (id < 0 || id >= eng->q35.capacity || !eng->slots[id].used) return -1;
        // Advancing the same stateful slot twice in one launch races its GDN/conv/KV writes.
        for (int p = 0; p < r; p++) if (slots[p] == id) return -1;
        gemma4_seq *s = &eng->slots[id];
        if (s->n_tokens >= eng->q35.maxctx) { if (out_tokens) out_tokens[r] = -1; continue; }
        if (q35_slot_kv_reserve(eng, id, s->n_tokens + 1) != 0) return -1;
        slv[Bv] = s; positions[Bv] = s->n_tokens; in2[Bv] = in_tokens[r]; rowmap[Bv] = r; Bv++;
    }
    if (Bv == 0) return 0;
    cudaStream_t st = eng->stream;
    const int splice = eng->q35.gpu_splice_enabled;
    if (splice) {
        // S2a — seed persistent per-slot state from this step's host inputs, device-side, then the
        // in-graph splice re-derives (d_sb[0], d_ms_pos) from slot state. The row→slot map upload
        // happens inside qwen35_ms_run. Byte-identical to the host-copy path; enables graph
        // self-chaining (writeback→splice) for a future zero-sync draft loop.
        int h_slot2[GEMMA4_MAX_SEQS];
        for (int v = 0; v < Bv; v++) h_slot2[v] = (int)(slv[v] - eng->slots);
        cudaMemcpyAsync(eng->q35.pf_tok, in2, (size_t)Bv*sizeof(int32_t), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(eng->q35.pf_pos, positions, (size_t)Bv*sizeof(int), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(workspace_data<int>(eng->q35.routing_workspace), h_slot2,
                        (size_t)Bv*sizeof(int), cudaMemcpyHostToDevice, st);
        qwen35_seed_slot_state_kernel<<<((unsigned)Bv+255)/256,256,0,st>>>(
            eng->q35.d_slot_tok, eng->q35.d_slot_pos, eng->q35.pf_tok, eng->q35.pf_pos,
            workspace_data<int>(eng->q35.routing_workspace), Bv);
    }
    qwen35_ms_run(eng, slv, in2, positions, Bv, /*want_argmax=*/1, /*use_graph=*/1, splice);
    int32_t outs[GEMMA4_MAX_SEQS];
    cudaMemcpyAsync(outs, eng->d_ms_outtok, (size_t)Bv*sizeof(int32_t), cudaMemcpyDeviceToHost, eng->stream);
    cudaStreamSynchronize(eng->stream);
    { cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
          fprintf(stderr, "fucina: qwen35_step_batch CUDA error (B=%d): %s\n", Bv, cudaGetErrorString(e));
          return -1;
      } }
    for (int v = 0; v < Bv; v++) {
        slv[v]->n_tokens = positions[v] + 1;
        slv[v]->n_sampled++;
        if (out_tokens) out_tokens[rowmap[v]] = outs[v];
    }
    return 0;
}

// ─── P0 (S1a): lossless GDN snapshot / rewind / commit for DFlash (1+K) verification ─────
//
// The hybrid slot's mutable decode state is (a) the FIXED-size GDN delta-rule matrix S + causal
// conv ring for every LINEAR layer, laid out as ONE contiguous recurrent_slab[slot]; and (b) the
// FULL-layer K/V cache, which is ABSOLUTE-POSITION-indexed. A DFlash verify pass advances the live
// slot through (1+K) candidate tokens to score them, but only the first j are accepted. Losslessly
// continuing requires the slot to end up EXACTLY as if only those j tokens had been decoded one at
// a time.
//
// Design (deterministic BY CONSTRUCTION): snapshot copies recurrent_slab[slot] byte-for-byte and
// records the absolute token count n0. commit(accepted, j) restores that snapshot, resets
// n_tokens=n0, then REPLAYS the j accepted tokens through the ordinary single-token decode path
// (qwen35_step_batch). Because commit literally re-executes the standard sequential decode, the
// resulting GDN/conv state is bit-identical to j one-by-one decodes, and the next-token logits are
// produced by the same kernels in the same order. The FULL-layer K/V cache needs no snapshot: the
// replay overwrites positions [n0, n0+j) and attention never reads past n_tokens, so the stale
// speculative positions [n0+j, n0+K) are inert (same invariant as qwen35_slot_reset). rewind is
// commit with j=0: pure restore to the pre-verify snapshot.
//
// Reuses the S2 persistent per-slot arenas; adds only one device scratch slab per slot (lazy).
// No host allocation or STL in the hot path; the replay loop is a fixed j<=K count of int steps.

static int q35_gdn_snapshot(gemma4_engine_t *eng, int slot) {
    if (!eng || eng->cfg.arch != GEMMA4_ARCH_QWEN3_5) return -1;
    if (slot < 0 || slot >= eng->q35.capacity || !eng->slots[slot].used) return -1;
    if (!eng->q35.slot_allocated[slot]) return -1;
    const size_t bytes = eng->q35.per_slot_recurrent_bytes;
    if (bytes == 0 || !eng->q35.recurrent_slab[slot]) return -1;
    if (!eng->q35.gdn_snap_slab[slot]) {
        if (cudaMalloc(&eng->q35.gdn_snap_slab[slot], bytes) != cudaSuccess) {
            cudaGetLastError(); return -1;
        }
    }
    cudaStream_t st = eng->stream;
    cudaMemcpyAsync(eng->q35.gdn_snap_slab[slot], eng->q35.recurrent_slab[slot],
                    bytes, cudaMemcpyDeviceToDevice, st);
    if (cudaStreamSynchronize(st) != cudaSuccess) return -1;
    eng->q35.gdn_snap_ntokens[slot] = eng->slots[slot].n_tokens;
    return 0;
}

// Restore the recurrent slab to the last snapshot and reset the slot's token count to n0. Does not
// touch the FULL-layer K/V arenas (absolute-position-indexed; overwritten on replay, never read
// stale). Internal helper shared by rewind and commit.
static int q35_gdn_restore_snapshot(gemma4_engine_t *eng, int slot) {
    if (slot < 0 || slot >= eng->q35.capacity || !eng->slots[slot].used) return -1;
    if (eng->q35.gdn_snap_ntokens[slot] < 0 || !eng->q35.gdn_snap_slab[slot]) return -1;
    if (!eng->q35.recurrent_slab[slot]) return -1;
    const size_t bytes = eng->q35.per_slot_recurrent_bytes;
    cudaStream_t st = eng->stream;
    cudaMemcpyAsync(eng->q35.recurrent_slab[slot], eng->q35.gdn_snap_slab[slot],
                    bytes, cudaMemcpyDeviceToDevice, st);
    if (cudaStreamSynchronize(st) != cudaSuccess) return -1;
    gemma4_seq *s = &eng->slots[slot];
    s->n_tokens  = eng->q35.gdn_snap_ntokens[slot];
    s->n_sampled = 0;
    s->mtp_h_valid = 0;
    return 0;
}

// commit(accepted, j): restore the pre-verify snapshot, then replay exactly the j accepted tokens
// through the standard decode path so the slot ends byte-identical to j sequential single-token
// decodes. `accepted[i]` is the input token fed at replay step i (i.e. the token decoded at
// absolute position n0+i). Optionally returns each replay step's next-token argmax in out_next.
// j==0 is a pure rewind. Returns 0 on success. The live snapshot is consumed (ntokens reset to -1)
// so a stale snapshot can never be silently reused.
static int q35_gdn_commit(gemma4_engine_t *eng, int slot, const int32_t *accepted, int j,
                          int32_t *out_next) {
    if (!eng || eng->cfg.arch != GEMMA4_ARCH_QWEN3_5) return -1;
    if (slot < 0 || slot >= eng->q35.capacity || !eng->slots[slot].used) return -1;
    if (j < 0) return -1;
    if (q35_gdn_restore_snapshot(eng, slot) != 0) return -1;
    for (int i = 0; i < j; i++) {
        int sl = slot; int32_t in = accepted[i], out = -1;
        if (qwen35_step_batch(eng, &sl, &in, 1, &out) != 0) {
            eng->q35.gdn_snap_ntokens[slot] = -1;
            return -1;
        }
        if (out_next) out_next[i] = out;
    }
    eng->q35.gdn_snap_ntokens[slot] = -1;
    return 0;
}

// Lossless FAST commit: replay the j accepted tokens in ONE weight pass (batched projections) with
// the GDN/conv recurrence run token-sequentially via the DECODE kernels (use_decode_gdn=1), so the
// committed per-slot state is byte-identical to j sequential decodes -- but ~1 weight read instead
// of j. want_logits=2 captures each committed row's argmax (the decode-body correction) in one pass.
// out_argmax[i] = greedy token after committing accepted[i]. Replaces the j-fold q35_gdn_commit loop.
static int q35_gdn_commit_fast(gemma4_engine_t *eng, int slot, const int32_t *accepted, int j,
                               int32_t *out_argmax) {
    if (!eng || eng->cfg.arch != GEMMA4_ARCH_QWEN3_5) return -1;
    if (slot < 0 || slot >= eng->q35.capacity || !eng->slots[slot].used || j < 0) return -1;
    if (q35_gdn_restore_snapshot(eng, slot) != 0) return -1;
    if (j == 0) { eng->q35.gdn_snap_ntokens[slot] = -1; return 0; }
    gemma4_seq *s = &eng->slots[slot];
    int base = s->n_tokens;
    if (base + j > eng->q35.maxctx) { eng->q35.gdn_snap_ntokens[slot] = -1; return -1; }
    if (q35_slot_kv_reserve(eng, slot, base + j) != 0) { eng->q35.gdn_snap_ntokens[slot] = -1; return -1; }
    cudaStream_t st = eng->stream;
    int32_t *d_tok = workspace_data<int32_t>(eng->q35.prefill_token_workspace);
    int     *d_pos = workspace_data<int>(eng->q35.prefill_position_workspace);
    int     *d_slot = workspace_data<int>(eng->q35.routing_workspace);
    int h_pos[GEMMA4_MAX_SEQS], h_slot[GEMMA4_MAX_SEQS];
    for (int t = 0; t < j; t++) { h_pos[t] = base + t; h_slot[t] = slot; }
    cudaMemcpyAsync(d_tok, accepted, (size_t)j*sizeof(int32_t), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_pos, h_pos, (size_t)j*sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_slot, h_slot, (size_t)j*sizeof(int), cudaMemcpyHostToDevice, st);
    qwen35_prefill_chunk_body_ex(eng, slot, base, j, out_argmax ? 2 : 0, /*use_decode_gdn=*/1, st);
    if (out_argmax)
        cudaMemcpyAsync(out_argmax, eng->d_ms_outtok, (size_t)j*sizeof(int32_t), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    if (cudaGetLastError() != cudaSuccess) { eng->q35.gdn_snap_ntokens[slot] = -1; return -1; }
    s->n_tokens = base + j;
    eng->q35.gdn_snap_ntokens[slot] = -1;
    return 0;
}

// ─── S1a P4: target (1+K) verify forward with all-row argmax capture + GDN rollback ─────
//
// Runs the target over a (1+K)-token block appended at the slot's current position, capturing the
// greedy argmax of EVERY row (each row t predicts the token after block position t), then rolls the
// GDN/conv recurrent state back to the pre-verify snapshot so the caller can commit only the
// accepted prefix via q35_gdn_commit. The block is [in_tokens[0..K]] (bonus/last-accepted token +
// K draft tokens). out_argmax receives T = 1+K per-row argmaxes. Because the chunk body advances
// state with the SAME kernels as sequential decode and captures per-row argmax with the SAME head,
// out_argmax[t] equals what a sequential decode would have produced at that position — the greedy
// verify contract. This does not commit; the caller applies the accept decision + commit.
static int qwen35_dflash_verify_block_ex(gemma4_engine_t *eng, int slot, const int32_t *block, int T,
                                         int32_t *out_argmax, float *out_logits) {
    if (!eng || !eng->loaded || eng->cfg.arch != GEMMA4_ARCH_QWEN3_5) return -1;
    if (slot < 0 || slot >= eng->q35.capacity || !eng->slots[slot].used) return -1;
    if (T <= 0 || T > GEMMA4_MAX_SEQS) return -1;
    if (ensure_q35_scratch(eng) != 0) return -1;
    gemma4_seq *s = &eng->slots[slot];
    int base = s->n_tokens;
    if (base + T > eng->q35.maxctx) return -1;
    if (q35_slot_kv_reserve(eng, slot, base + T) != 0) return -1;
    // Snapshot GDN state so we can roll back after scoring the speculative block.
    if (q35_gdn_snapshot(eng, slot) != 0) return -1;
    cudaStream_t st = eng->stream;
    // Upload the block tokens/positions into the prefill token/position workspace + slot map.
    int32_t *d_tok = workspace_data<int32_t>(eng->q35.prefill_token_workspace);
    int     *d_pos = workspace_data<int>(eng->q35.prefill_position_workspace);
    int     *d_slot = workspace_data<int>(eng->q35.routing_workspace);
    int h_pos[GEMMA4_MAX_SEQS], h_slot[GEMMA4_MAX_SEQS];
    for (int t = 0; t < T; t++) { h_pos[t] = base + t; h_slot[t] = slot; }
    cudaMemcpyAsync(d_tok, block, (size_t)T*sizeof(int32_t), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_pos, h_pos, (size_t)T*sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_slot, h_slot, (size_t)T*sizeof(int), cudaMemcpyHostToDevice, st);
    // Score the block (advances state through all T rows), capturing all-row argmax (want_logits=2).
    qwen35_prefill_chunk_body(eng, slot, base, T, /*want_logits=*/2, st);
    if (out_argmax)
        cudaMemcpyAsync(out_argmax, eng->d_ms_outtok, (size_t)T*sizeof(int32_t), cudaMemcpyDeviceToHost, st);
    // Probabilistic path: also copy the per-row target logits [T, vocab] from d_sb[11] (device->
    // device: the caller passes a device buffer). Greedy callers pass NULL and pay nothing.
    if (out_logits)
        cudaMemcpyAsync(out_logits, eng->d_sb[11], (size_t)T*eng->cfg.vocab_size*sizeof(float),
                        cudaMemcpyDeviceToDevice, st);
    cudaStreamSynchronize(st);
    if (cudaGetLastError() != cudaSuccess) { eng->q35.gdn_snap_ntokens[slot] = -1; return -1; }
    // Roll GDN state back to the snapshot (n_tokens restored to base). The snapshot is LEFT LIVE so
    // a following q35_gdn_commit can replay the accepted prefix from it; a caller that only scores
    // (no commit) should rewind explicitly. keep_snapshot=1 preserves gdn_snap_ntokens.
    if (q35_gdn_restore_snapshot(eng, slot) != 0) return -1;
    return 0;
}
// Back-compat greedy wrapper: argmax only, no logit copy.
static int qwen35_dflash_verify_block(gemma4_engine_t *eng, int slot, const int32_t *block, int T,
                                      int32_t *out_argmax) {
    return qwen35_dflash_verify_block_ex(eng, slot, block, T, out_argmax, nullptr);
}

// ─── S1a P4: one greedy DFlash step (verify + accept + commit) ───────────────────────
//
// One DFlash speculative step, greedy, LOSSLESS by construction and independent of draft content.
// Inputs: the slot at position `base` with `bonus` the token to decode next (last step's emitted
// token, not yet fed), plus K candidate draft tokens. Steps:
//   1. verify block [bonus, d1..dK] -> per-row argmax am[0..K] (state rolled back to base).
//   2. greedy accept: j = number of leading di == am[i-1]; on first mismatch (or all K) the emitted
//      correction is am[j] (the true greedy token at that position).
//   3. commit: replay [bonus, d1..dj] (1+j tokens) into state so the slot advances exactly as
//      sequential greedy decode would have. am[j] becomes the NEXT step's bonus.
// Emits out_emit[0..nemit) = [d1..dj, am[j]] (the j accepted greedy tokens + the correction), all
// equal to greedy decode. Returns the next bonus (am[j]) via *next_bonus, nemit via *out_n. Because
// commit replays through the standard decode path, greedy output is byte-identical to non-spec
// greedy decode for ANY drafts (wrong drafts only lower j). K<=GEMMA4_MAX_SEQS-1.
static int qwen35_dflash_greedy_step(gemma4_engine_t *eng, int slot, int32_t bonus,
                                     const int32_t *draft, int K,
                                     int32_t *out_emit, int *out_n, int32_t *next_bonus) {
    if (!eng || eng->cfg.arch != GEMMA4_ARCH_QWEN3_5 || K < 0 || K + 1 > GEMMA4_MAX_SEQS) return -1;
    int T = 1 + K;
    int32_t block[GEMMA4_MAX_SEQS], am[GEMMA4_MAX_SEQS];
    block[0] = bonus;
    for (int i = 0; i < K; i++) block[1 + i] = draft[i];
    if (qwen35_dflash_verify_block(eng, slot, block, T, am) != 0) return -1;
    // Greedy accept: di (block[1+i]) is accepted iff it equals am[i] (row i predicts the token after
    // block position i). Accept the leading run.
    // Provisional accept using the verify chunk-body argmax (a FAST filter to bound how many tokens
    // to replay). The AUTHORITATIVE greedy decision is the DECODE body, applied below.
    int jhat = 0;
    while (jhat < K && block[1 + jhat] == am[jhat]) jhat++;
    // Replay [bonus, d1..d_jhat] through the standard DECODE path, capturing each step's argmax.
    // The verify chunk body and the decode body can produce subtly different logits for the same
    // tokens (different kernels/accumulation order), so the EMITTED tokens must be validated against
    // the decode body, never the chunk-body argmax am[] (trusting am[] drifts from true sequential
    // decode over many steps -- a real losslessness bug on some sequences).
    // The emitted tokens must equal what the DECODE body produces (the chunk-body am[] argmax can
    // diverge from the decode body at some positions -- a real losslessness bug), so the accept
    // length and correction are re-derived from q35_gdn_commit's per-step DECODE-body argmax
    // out_next[]. (q35_gdn_commit_fast produces byte-identical STATE in ~one weight pass and is a
    // validated primitive, but its want_logits argmax uses the chunk-body head and is NOT decode-
    // identical at all positions, so it is not used for the emitted correction here.)
    int32_t commit_toks[GEMMA4_MAX_SEQS], out_next[GEMMA4_MAX_SEQS];
    commit_toks[0] = bonus;
    for (int i = 0; i < jhat; i++) commit_toks[1 + i] = block[1 + i];
    if (q35_gdn_commit(eng, slot, commit_toks, 1 + jhat, out_next) != 0) return -1;
    int j = 0;
    while (j < jhat && block[1 + j] == out_next[j]) j++;
    if (j < jhat) {
        for (int i = 0; i < j; i++) commit_toks[1 + i] = block[1 + i];
        if (q35_gdn_commit(eng, slot, commit_toks, 1 + j, out_next) != 0) return -1;
    }
    int32_t correction = out_next[j];   // decode-body greedy token at position base+j (next bonus)
    // Emit the j decode-accepted drafts + the decode-derived correction (all == plain greedy decode).
    int n = 0;
    for (int i = 0; i < j; i++) out_emit[n++] = block[1 + i];
    out_emit[n++] = correction;
    *out_n = n;
    *next_bonus = correction;
    return 0;
}

// ── M4 gate self-test (qwen35) ─────────────────────────────────────────────────────────
// Drives the qwen35 continuous-batching ABI (seq_add prefill + step_batch decode) and asserts
//   (1) B-row batched decode (graph ON) is BIT-IDENTICAL per row to that row run alone B=1
//       (graph OFF) — the batch self-test invariant (row independence + graph correctness);
//   (2) graph-ON == graph-OFF for the B-row batch (CUDA-graph determinism);
//   (3) the batched path reproduces the proven M3 single-seq forward (qwen35_forward_greedy)
//       — the France→Paris 8/8 continuation — so the batch is not merely self-consistent.
// Returns 0 on PASS. Prints a verbatim, paste-able report.
extern "C" int qwen35_batch_selftest(gemma4_engine_t *eng) {
    if (!eng || eng->cfg.arch != GEMMA4_ARCH_QWEN3_5) {
        fprintf(stderr, "qwen35_batch_selftest: engine is not qwen35\n"); return 1;
    }
    const int NSEQ = 3, KSTEP = 24, MAXP = 11;
    // RAGGED prompt lengths (5/8/11) so the three rows sit at DIFFERENT absolute positions in
    // every batched step — the genuine continuous-batching case (sequences that joined at
    // different times). With equal lengths the rows happen to share positions, which masks any
    // per-row position bug (d_ms_pos / rowslot / FULL-cache write offset / GDN-state slot key).
    // seq 0 = "The capital of France is" (the M3 reference prompt, kept at 5 for the oracle);
    // seq 1,2 = arbitrary in-vocab ids of length 8 and 11.
    const int NPq[NSEQ] = { 5, 8, 11 };
    int32_t prompt[NSEQ][MAXP] = {
        { 760, 6511, 314, 9338, 369 },
        { 785, 6722, 315, 9625, 374, 1024, 2048, 4096 },
        { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100 },
    };
    int32_t ref[NSEQ][KSTEP], bat[NSEQ][KSTEP], boff[NSEQ][KSTEP];

    // (A) reference: each seq ALONE, B=1, graph OFF.
    eng->q35.graph_enabled = 0;
    for (int q = 0; q < NSEQ; q++) {
        int32_t first = 0;
        int slot = gemma4_engine_seq_add(eng, prompt[q], NPq[q], &first, 0.f, 0, 0.f, 0.f, 0);
        if (slot < 0) { fprintf(stderr, "qwen35_batch_selftest: seq_add(ref) failed\n"); return 1; }
        int32_t tok = first;
        for (int k = 0; k < KSTEP; k++) {
            ref[q][k] = tok; int32_t nxt = 0; int sl = slot;
            if (gemma4_engine_step_batch(eng, &sl, &tok, 1, &nxt) != 0) {
                fprintf(stderr, "qwen35_batch_selftest: step(ref) failed\n");
                gemma4_engine_seq_remove(eng, slot); return 1;
            }
            tok = nxt;
        }
        gemma4_engine_seq_remove(eng, slot);
    }

    // (B) batched: all NSEQ together, B=NSEQ, graph ON.
    eng->q35.graph_enabled = 1;
    {
        int slot[NSEQ]; int32_t cur[NSEQ];
        for (int q = 0; q < NSEQ; q++) {
            int32_t first = 0;
            slot[q] = gemma4_engine_seq_add(eng, prompt[q], NPq[q], &first, 0.f, 0, 0.f, 0.f, 0);
            if (slot[q] < 0) { fprintf(stderr, "qwen35_batch_selftest: seq_add(batch) failed\n"); return 1; }
            cur[q] = first;
        }
        for (int k = 0; k < KSTEP; k++) {
            int32_t nxt[NSEQ];
            for (int q = 0; q < NSEQ; q++) bat[q][k] = cur[q];
            if (gemma4_engine_step_batch(eng, slot, cur, NSEQ, nxt) != 0) {
                fprintf(stderr, "qwen35_batch_selftest: step(batch) failed\n");
                for (int q = 0; q < NSEQ; q++) gemma4_engine_seq_remove(eng, slot[q]); return 1;
            }
            for (int q = 0; q < NSEQ; q++) cur[q] = nxt[q];
        }
        for (int q = 0; q < NSEQ; q++) gemma4_engine_seq_remove(eng, slot[q]);
    }

    // (C) batched again, B=NSEQ, graph OFF (graph determinism check).
    eng->q35.graph_enabled = 0;
    {
        int slot[NSEQ]; int32_t cur[NSEQ];
        for (int q = 0; q < NSEQ; q++) {
            int32_t first = 0;
            slot[q] = gemma4_engine_seq_add(eng, prompt[q], NPq[q], &first, 0.f, 0, 0.f, 0.f, 0);
            if (slot[q] < 0) { fprintf(stderr, "qwen35_batch_selftest: seq_add(boff) failed\n"); return 1; }
            cur[q] = first;
        }
        for (int k = 0; k < KSTEP; k++) {
            int32_t nxt[NSEQ];
            for (int q = 0; q < NSEQ; q++) boff[q][k] = cur[q];
            if (gemma4_engine_step_batch(eng, slot, cur, NSEQ, nxt) != 0) {
                fprintf(stderr, "qwen35_batch_selftest: step(boff) failed\n");
                for (int q = 0; q < NSEQ; q++) gemma4_engine_seq_remove(eng, slot[q]); return 1;
            }
            for (int q = 0; q < NSEQ; q++) cur[q] = nxt[q];
        }
        for (int q = 0; q < NSEQ; q++) gemma4_engine_seq_remove(eng, slot[q]);
    }
    eng->q35.graph_enabled = 1;

    // ── compare ──
    int indep_ok = 1, graph_ok = 1;
    for (int q = 0; q < NSEQ; q++) {
        int agree = 0, gagree = 0, fm = -1;
        for (int k = 0; k < KSTEP; k++) {
            if (bat[q][k] == ref[q][k]) agree++; else if (fm < 0) fm = k;
            if (bat[q][k] == boff[q][k]) gagree++;
        }
        if (agree  != KSTEP) indep_ok = 0;
        if (gagree != KSTEP) graph_ok = 0;
        fprintf(stderr,
            "qwen35 M4 seq %d (prompt=%d, decode pos %d..%d): B=%d(graph) vs B=1(per-kernel) "
            "%d/%d bit-identical%s%s; graph-on vs graph-off %d/%d\n",
            q, NPq[q], NPq[q], NPq[q] + KSTEP - 1, NSEQ, agree, KSTEP,
            (fm >= 0) ? "  first-mismatch@" : "", "", gagree, KSTEP);
        if (fm >= 0)
            fprintf(stderr, "qwen35 M4   seq %d first mismatch step %d (B=1 %d vs B=%d %d)\n",
                    q, fm, ref[q][fm], NSEQ, bat[q][fm]);
    }

    // (D) M3 cross-check: the batched decode of seq 0 must reproduce qwen35_forward_greedy.
    const int NGEN = 12, GATE = 8;
    int32_t m3[NGEN] = {0};
    int m3_ok = (qwen35_forward_greedy(eng, prompt[0], NPq[0], m3, NGEN) == 0);
    int m3_agree = 0;
    if (m3_ok) {
        // ref[0] = [first, then KSTEP-1 decoded]; m3 = [first, then NGEN-1 decoded]. Compare GATE.
        for (int k = 0; k < GATE && k < KSTEP; k++) if (ref[0][k] == m3[k]) m3_agree++;
        fprintf(stderr, "qwen35 M4 seq 0 vs M3 forward (France->Paris): ");
        for (int k = 0; k < GATE; k++) fprintf(stderr, "%d%s", ref[0][k], (k<GATE-1)?",":"");
        fprintf(stderr, "  (M3: ");
        for (int k = 0; k < GATE; k++) fprintf(stderr, "%d%s", m3[k], (k<GATE-1)?",":"");
        fprintf(stderr, ")  %d/%d\n", m3_agree, GATE);
    } else {
        fprintf(stderr, "qwen35 M4 seq 0 vs M3 forward: qwen35_forward_greedy failed\n");
    }

    // (E) Sampling + mixed-batch gate: fixed per-sequence seeds must produce the same tokens when
    // each row runs alone or beside rows with different temperatures. Row 0 is greedy, which also
    // proves the sampler's temp<=0 branch matches the CUDA-graph argmax path.
    const int SK = 4;
    const float ST[NSEQ] = {0.f, 0.8f, 1.1f};
    int32_t sref[NSEQ][SK], sbat[NSEQ][SK];
    for (int q = 0; q < NSEQ; q++) {
        int32_t cur = 0; uint64_t seed = 0x12345678ULL + q;
        int sl = gemma4_engine_seq_add(eng, prompt[q], NPq[q], &cur,
                                       ST[q], 32, 0.9f, 0.f, seed);
        if (sl < 0) return 1;
        for (int k = 0; k < SK; k++) {
            sref[q][k] = cur; int32_t next = 0;
            if (gemma4_engine_step_batch(eng, &sl, &cur, 1, &next) != 0) return 1;
            cur = next;
        }
        gemma4_engine_seq_remove(eng, sl);
    }
    {
        int sl[NSEQ]; int32_t cur[NSEQ];
        for (int q = 0; q < NSEQ; q++) {
            uint64_t seed = 0x12345678ULL + q;
            sl[q] = gemma4_engine_seq_add(eng, prompt[q], NPq[q], &cur[q],
                                          ST[q], 32, 0.9f, 0.f, seed);
            if (sl[q] < 0) return 1;
        }
        for (int k = 0; k < SK; k++) {
            int32_t next[NSEQ];
            for (int q = 0; q < NSEQ; q++) sbat[q][k] = cur[q];
            if (gemma4_engine_step_batch(eng, sl, cur, NSEQ, next) != 0) return 1;
            for (int q = 0; q < NSEQ; q++) cur[q] = next[q];
        }
        for (int q = 0; q < NSEQ; q++) gemma4_engine_seq_remove(eng, sl[q]);
    }
    int sample_ok = 1;
    for (int q = 0; q < NSEQ; q++) for (int k = 0; k < SK; k++)
        if (sref[q][k] != sbat[q][k]) sample_ok = 0;

    // (F) S2a self-chaining gate: seed the persistent per-slot state ONCE, then replay the captured
    // greedy decode graph KSTEP times with NO host token feedback between steps — the graph's
    // in-body writeback→splice must advance (token, position) device-side. The produced tokens
    // must byte-match the host-fed batched reference `bat` from section (B). This proves the whole
    // decode step is a self-contained replayable graph (the S1 zero-sync draft-loop prerequisite).
    // Only runs when GPU splicing + graphs are enabled (the default); otherwise trivially passes.
    int chain_ok = 1;
    if (eng->q35.gpu_splice_enabled && eng->q35.graph_enabled && !eng->q35.graph_failed) {
        int sl[NSEQ]; int32_t cur[NSEQ];
        for (int q = 0; q < NSEQ; q++) {
            sl[q] = gemma4_engine_seq_add(eng, prompt[q], NPq[q], &cur[q], 0.f, 0, 0.f, 0.f, 0);
            if (sl[q] < 0) { fprintf(stderr, "qwen35 M4 self-chain: seq_add failed\n"); return 1; }
        }
        cudaStream_t st = eng->stream;
        int h_slot[NSEQ]; int32_t h_tok[NSEQ]; int h_pos[NSEQ];
        for (int q = 0; q < NSEQ; q++) {
            h_slot[q] = sl[q]; h_tok[q] = cur[q]; h_pos[q] = eng->slots[sl[q]].n_tokens;
        }
        // Seed slot state + row→slot map ONCE, on the same stream as the graph.
        cudaMemcpyAsync(eng->q35.pf_tok, h_tok, (size_t)NSEQ*sizeof(int32_t), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(eng->q35.pf_pos, h_pos, (size_t)NSEQ*sizeof(int), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(workspace_data<int>(eng->q35.routing_workspace), h_slot,
                        (size_t)NSEQ*sizeof(int), cudaMemcpyHostToDevice, st);
        qwen35_seed_slot_state_kernel<<<1,256,0,st>>>(
            eng->q35.d_slot_tok, eng->q35.d_slot_pos, eng->q35.pf_tok, eng->q35.pf_pos,
            workspace_data<int>(eng->q35.routing_workspace), NSEQ);
        q35_graph_key key = q35_make_decode_key(NSEQ);
        cudaGraphExec_t exec = qwen35_ms_graph_ensure(eng, key);
        if (!exec) { fprintf(stderr, "qwen35 M4 self-chain: graph ensure failed\n"); chain_ok = 0; }
        int32_t chain[NSEQ][KSTEP];
        for (int k = 0; k < KSTEP && exec; k++) {
            // Row 0's produced token this step is the SEEDED input (the graph writes back AFTER
            // argmax), so read slot_tok BEFORE replay to capture step k's input == bat[q][k].
            int32_t seen[NSEQ];
            cudaMemcpyAsync(seen, eng->q35.d_slot_tok, (size_t)NSEQ*sizeof(int32_t),
                            cudaMemcpyDeviceToHost, st); // slot ids 0..NSEQ-1 are contiguous here
            cudaStreamSynchronize(st);
            for (int q = 0; q < NSEQ; q++) chain[q][k] = seen[sl[q]];
            if (cudaGraphLaunch(exec, st) != cudaSuccess) { chain_ok = 0; break; }
            cudaStreamSynchronize(st);
            // Keep host seq bookkeeping in step with the device-advanced positions.
            for (int q = 0; q < NSEQ; q++) { eng->slots[sl[q]].n_tokens++; eng->slots[sl[q]].n_sampled++; }
        }
        for (int q = 0; q < NSEQ && chain_ok; q++)
            for (int k = 0; k < KSTEP; k++)
                if (chain[q][k] != bat[q][k]) { chain_ok = 0; break; }
        for (int q = 0; q < NSEQ; q++) gemma4_engine_seq_remove(eng, sl[q]);
        fprintf(stderr, "qwen35 M4 self-chain (seed-once, %d graph replays, no host feedback): %s\n",
                KSTEP, chain_ok ? "PASS" : "FAIL");
    }

    int pass = indep_ok && graph_ok && m3_ok && (m3_agree == GATE) && sample_ok && chain_ok;
    fprintf(stderr,
        "qwen35 M4 batched-decode gate: row-independence=%s graph-on==off=%s "
        "M3-parity=%s(%d/%d) sampling=%s self-chain=%s — %s\n",
        indep_ok ? "PASS" : "FAIL", graph_ok ? "PASS" : "FAIL",
        (m3_ok && m3_agree == GATE) ? "PASS" : "FAIL", m3_agree, GATE,
        sample_ok ? "PASS" : "FAIL", chain_ok ? "PASS" : "FAIL", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}


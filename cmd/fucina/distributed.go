package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/hikmaai-io/fucina/internal/dist"
	"github.com/hikmaai-io/fucina/internal/engine/cuda"
	"github.com/hikmaai-io/fucina/internal/sampler"
	"github.com/hikmaai-io/fucina/internal/session"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

func parseLayerRange(s string) (int, int, error) {
	parts := strings.Split(s, ":")
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("layer range %q must be lo:hi", s)
	}
	lo, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil {
		return 0, 0, fmt.Errorf("layer range %q: %w", s, err)
	}
	hi, err := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err != nil {
		return 0, 0, fmt.Errorf("layer range %q: %w", s, err)
	}
	if lo < 0 || hi <= lo {
		return 0, 0, fmt.Errorf("invalid layer range [%d,%d)", lo, hi)
	}
	return lo, hi, nil
}

func distributedHello(model string, info cuda.ShardInfo, lo, hi int, final bool) (dist.Hello, error) {
	hash, err := session.HashModelConfig(model)
	if err != nil {
		return dist.Hello{}, err
	}
	h := dist.Hello{
		Version: dist.Version, ConfigHash: hash, LayerLo: lo, LayerHi: hi,
		Hidden: info.Hidden, DType: dist.DTypeF32, Final: final,
	}
	if err := dist.ValidateHello(h); err != nil {
		return dist.Hello{}, err
	}
	return h, nil
}

func runDistributedWorker(eng *cuda.Engine, args CLIArgs) {
	lo, hi, err := parseLayerRange(args.DistLayers)
	if err != nil {
		log.Fatalf("fucina: %v", err)
	}
	runner, err := cuda.NewCUDAShardRunner(eng, lo, hi, args.DistFinal)
	if err != nil {
		log.Fatalf("fucina: distributed worker: %v", err)
	}
	hello, err := distributedHello(args.ModelPath, eng.ShardInfo(), lo, hi, args.DistFinal)
	if err != nil {
		log.Fatalf("fucina: distributed worker identity: %v", err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	log.Printf("fucina: Phase-E worker listening on %s, layers [%d,%d), final=%v, config=%016x",
		args.DistListen, lo, hi, args.DistFinal, hello.ConfigHash)
	if err := dist.ListenAndServeContext(ctx, args.DistListen,
		&dist.Worker{Hello: hello, Runner: runner, Final: args.DistFinal}); err != nil && ctx.Err() == nil {
		log.Fatalf("fucina: distributed worker: %v", err)
	}
}

func runDistributedOneShot(eng *cuda.Engine, tok *tokenizer.Tokenizer, args CLIArgs) {
	lo, hi, err := parseLayerRange(args.DistLayers)
	if err != nil {
		log.Fatalf("fucina: %v", err)
	}
	if lo != 0 {
		log.Fatalf("fucina: coordinator range must start at layer 0, got [%d,%d)", lo, hi)
	}
	info := eng.ShardInfo()
	if hi >= info.Layers {
		log.Fatalf("fucina: coordinator must leave at least one final worker shard (local [%d,%d), model layers=%d)", lo, hi, info.Layers)
	}
	local, err := cuda.NewCUDAShardRunner(eng, lo, hi, false)
	if err != nil {
		log.Fatalf("fucina: distributed coordinator: %v", err)
	}
	mine, err := distributedHello(args.ModelPath, info, lo, hi, false)
	if err != nil {
		log.Fatalf("fucina: distributed coordinator identity: %v", err)
	}

	addresses := strings.Split(args.DistWorkers, ",")
	hops := make([]*dist.Hop, 0, len(addresses))
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	for i, raw := range addresses {
		addr := strings.TrimSpace(raw)
		if addr == "" {
			log.Fatalf("fucina: empty worker address at index %d", i)
		}
		hop, err := dist.DialTCP(ctx, addr, mine)
		if err != nil {
			for _, h := range hops {
				_ = h.Close()
			}
			log.Fatalf("fucina: dial distributed worker %d (%s): %v", i, addr, err)
		}
		if err := hop.Ping(); err != nil {
			_ = hop.Close()
			log.Fatalf("fucina: ping distributed worker %d (%s): %v", i, addr, err)
		}
		peer := hop.Peer()
		if peer.Final && i != len(addresses)-1 {
			_ = hop.Close()
			log.Fatalf("fucina: worker %d (%s) is final but later workers were configured", i, addr)
		}
		log.Printf("fucina: Phase-E hop %d %s layers [%d,%d) final=%v ready", i, addr, peer.LayerLo, peer.LayerHi, peer.Final)
		hops = append(hops, hop)
		mine = peer
	}
	defer func() {
		for _, h := range hops {
			_ = h.Close()
		}
	}()
	if len(hops) == 0 || !hops[len(hops)-1].Peer().Final || hops[len(hops)-1].Peer().LayerHi != info.Layers {
		log.Fatalf("fucina: distributed route ends at layer %d, model has %d",
			mine.LayerHi, info.Layers)
	}

	prompt := args.Prompt
	if args.PromptFile != "" {
		b, err := os.ReadFile(args.PromptFile)
		if err != nil {
			log.Fatalf("fucina: cannot read prompt file: %v", err)
		}
		prompt = string(b)
	}
	if prompt == "" {
		log.Fatal("fucina: distributed coordinator currently supports one-shot mode; use -p or -f")
	}
	if args.System != "" {
		prompt = fmt.Sprintf("System: %s\n\n%s", args.System, prompt)
	}
	tokens := tok.Encode(prompt, true, false)
	if len(tokens) == 0 {
		log.Fatal("fucina: distributed prompt tokenized to zero tokens")
	}
	if args.Spec {
		log.Printf("fucina: Phase-E uses exact token-sequential forwarding; speculative decode is disabled")
	}

	pipeline := &dist.Pipeline{Local: local, Hops: hops}
	const seqID uint32 = 1
	defer func() {
		if err := pipeline.Reset(seqID); err != nil {
			log.Printf("fucina: distributed reset: %v", err)
		}
	}()

	start := time.Now()
	var logits []float32
	for pos, token := range tokens {
		hidden, err := eng.EmbedShardToken(token)
		if err != nil {
			log.Fatalf("fucina: distributed embed: %v", err)
		}
		out, err := pipeline.Forward(seqID, uint32(pos), 1, hidden)
		if err != nil {
			log.Fatalf("fucina: distributed prefill at position %d: %v", pos, err)
		}
		logits, err = cuda.BytesToFloat32(out)
		if err != nil || len(logits) != info.Vocab {
			log.Fatalf("fucina: distributed logits: bytes decode error=%v width=%d want=%d", err, len(logits), info.Vocab)
		}
	}
	prefillElapsed := time.Since(start)
	log.Printf("fucina: distributed token-sequential prefill %d tokens in %.2fs (%.1f tok/s)",
		len(tokens), prefillElapsed.Seconds(), float64(len(tokens))/prefillElapsed.Seconds())

	nToGenerate := args.Predict
	if nToGenerate < 0 {
		nToGenerate = 1 << 20
	}
	rng := newRNG(args.Seed)
	past := append([]int32(nil), tokens...)
	fmt.Print(prompt)
	genStart := time.Now()
	generated := 0
	for generated < nToGenerate {
		next, err := sampler.Sample(logits, samplerParams(args), rng, past)
		if err != nil || tok.IsStop(next) {
			break
		}
		fmt.Print(tok.Decode([]int32{next}))
		past = append(past, next)
		generated++
		hidden, err := eng.EmbedShardToken(next)
		if err != nil {
			log.Printf("fucina: distributed embed failed: %v", err)
			break
		}
		out, err := pipeline.Forward(seqID, uint32(len(tokens)+generated-1), 1, hidden)
		if err != nil {
			log.Printf("fucina: distributed decode failed: %v", err)
			break
		}
		logits, err = cuda.BytesToFloat32(out)
		if err != nil {
			log.Printf("fucina: distributed logits failed: %v", err)
			break
		}
	}
	fmt.Println()
	elapsed := time.Since(genStart)
	tps := 0.0
	if elapsed > 0 {
		tps = float64(generated) / elapsed.Seconds()
	}
	log.Printf("fucina: distributed generated %d tokens in %.2fs (%.1f tok/s)", generated, elapsed.Seconds(), tps)
}

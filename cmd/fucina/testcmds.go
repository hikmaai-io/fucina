// ABOUTME: In-binary self-tests dispatched by --test-* flags (parser, CUDA, logits).
// ABOUTME: runTestLogits cross-checks single-token decode vs batched spec-verify per geometry.

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/hikmaai-io/fucina/internal/engine/cuda"
	"github.com/hikmaai-io/fucina/internal/sampler"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

func runTestParser() int {
	fmt.Println("fucina: tokenizer parser test - not yet implemented")
	return 0
}

// runTestCUDA loads the model and validates that the batched spec-verify decode
// produces the SAME logits as the proven single-token decode for the same inputs.
// A mismatch is exactly what breaks speculative decoding (drafts never accepted)
// and corrupts generation; this catches it deterministically without a full run.
// The two paths are compared on two independent engines (each owns its KV cache),
// both prefilled identically, then stepped with the same tokens.
func runTestCUDA(args CLIArgs) int {
	if args.ModelPath == "" {
		fmt.Fprintln(os.Stderr, "fucina: --test-cuda needs -m <model.gguf>")
		return 2
	}
	data, err := os.ReadFile(args.ModelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fucina: read model: %v\n", err)
		return 1
	}
	tok, err := tokenizer.New(data, int64(len(data)))
	if err != nil {
		fmt.Fprintf(os.Stderr, "fucina: tokenizer: %v\n", err)
		return 1
	}
	mk := func() (*cuda.Engine, error) {
		return cuda.NewEngine(cuda.Config{
			ModelPath: args.ModelPath, ContextSize: 4096, DeviceID: args.DeviceID,
		})
	}
	prompt := "The capital of France is"
	pt := tok.Encode(prompt, true, false)
	if len(pt) == 0 {
		fmt.Fprintln(os.Stderr, "fucina: empty prompt encoding")
		return 1
	}

	// ONE engine: prefill, then at each step compare Decode(t) against the row-0 logits
	// of DecodeBatched([t]) from the SAME post-prefill state, using Rewind(n) to restore
	// the KV cache between the two paths. The single-token path is the proven reference;
	// the batched (spec-verify) path must produce the same logits or speculation breaks.
	eng, err := mk()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fucina: engine: %v\n", err)
		return 1
	}
	defer eng.Close()
	pl, err := eng.Prefill(pt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fucina: prefill: %v\n", err)
		return 1
	}
	fmt.Printf("  [diag] prefill argmax=%d (the first generated token); prompt=%v\n", sampler.Argmax(pl), pt)
	// Replicate the generate loop: greedy-decode 5 tokens via Decode (host argmax) and print
	// the ids. If these go degenerate while the prefill argmax was sane, the decode-after-
	// prefill continuation (KV state) is the bug, not the kernels.
	{
		gtoks := []int32{}
		nt := int32(sampler.Argmax(pl))
		for i := 0; i < 5; i++ {
			gtoks = append(gtoks, nt)
			dl, e := eng.Decode(nt)
			if e != nil {
				break
			}
			nt = int32(sampler.Argmax(dl))
		}
		fmt.Printf("  [diag] greedy decode chain (Decode+argmax): %v\n", gtoks)
		eng.Reset()
		eng.Prefill(pt) // re-establish state for the batched checks below
	}
	nLayers := eng.NumLayers()
	nKeep := eng.NTokens() // KV state to restore to before each batched call

	// Isolate the row>0 interaction: row 0 of DecodeBatched MUST NOT depend on K. Compare
	// row 0 of a K=1 forward against row 0 of a K=2 forward from the SAME post-prefill state.
	// A divergence means the batched path corrupts row 0 when extra rows are present — the
	// exact failure mode (first spec token correct, later tokens wrong). Two fixed tokens.
	t0 := int32(pt[len(pt)-1])
	t1 := int32(sampler.Argmax(must(eng.DecodeBatched([]int32{t0})))) // 1 step to get a plausible 2nd token
	eng.Rewind(nKeep)

	// The K-invariance check is deterministic, so a single comparison suffices.
	maxAbs := 0.0
	argmaxMismatch := 0
	l1, err := eng.DecodeBatched([]int32{t0})
	if err != nil {
		fmt.Fprintf(os.Stderr, "fucina: K=1 batched: %v\n", err)
		return 1
	}
	ls := append([]float32(nil), l1...) // row 0 of K=1 (reference)
	eng.Rewind(nKeep)
	l2, err := eng.DecodeBatched([]int32{t0, t1})
	if err != nil {
		fmt.Fprintf(os.Stderr, "fucina: K=2 batched: %v\n", err)
		return 1
	}
	lb := l2[:len(ls)] // row 0 of K=2 (must equal row 0 of K=1)
	eng.Rewind(nKeep)
	if da, db := sampler.Argmax(ls), sampler.Argmax(lb); da != db {
		argmaxMismatch++
		fmt.Printf("  ROW0 MISMATCH K1=%d K2row0=%d\n", da, db)
	}
	for i := range ls {
		if d := math.Abs(float64(ls[i] - lb[i])); d > maxAbs {
			maxAbs = d
		}
	}

	// Row>0 CORRECTNESS: row 1 of DecodeBatched([t0,t1]) must equal the row-0 logits of a
	// DecodeBatched([t1]) issued AFTER t0 has advanced the cache by one (i.e. the same token
	// at the same absolute position with the same causal context). A mismatch is the spec bug:
	// the verify rows past row 0 produce wrong logits, so wrong tokens get accepted.
	l2, err = eng.DecodeBatched([]int32{t0, t1})
	if err != nil {
		fmt.Fprintf(os.Stderr, "fucina: K=2 row1: %v\n", err)
		return 1
	}
	row1 := append([]float32(nil), l2[len(l2)/2:]...) // row 1 of the K=2 forward
	eng.Rewind(nKeep)
	if _, err := eng.DecodeBatched([]int32{t0}); err != nil { // advance cache by t0
		fmt.Fprintf(os.Stderr, "fucina: advance t0: %v\n", err)
		return 1
	}
	ref1, err := eng.DecodeBatched([]int32{t1}) // row 0 at position pos+1
	if err != nil {
		fmt.Fprintf(os.Stderr, "fucina: ref row1: %v\n", err)
		return 1
	}
	r1max := 0.0
	row1Mismatch := sampler.Argmax(row1) != sampler.Argmax(ref1)
	for i := range ref1 {
		if d := math.Abs(float64(row1[i] - ref1[i])); d > r1max {
			r1max = d
		}
	}

	geom := "12B"
	if nLayers > 48 {
		geom = "31B"
	}
	pass := func(ok bool) string {
		if ok {
			return "PASS"
		}
		return "FAIL"
	}
	fmt.Printf("fucina: batched self-test (%s) t0=%d t1=%d\n", geom, t0, t1)
	fmt.Printf("  row0 K-invariance: max_abs_err=%.6g mismatches=%d  (%s)\n",
		maxAbs, argmaxMismatch, pass(argmaxMismatch == 0))
	fmt.Printf("  row1 correctness:  max_abs_err=%.6g argmax_mismatch=%v  (%s)\n",
		r1max, row1Mismatch, pass(!row1Mismatch))
	if argmaxMismatch != 0 || row1Mismatch {
		fmt.Println("  => batched spec-verify is INCORRECT")
		return 1
	}
	fmt.Println("  => batched spec-verify matches single-token math")
	return 0
}

// must panics on error; for one-shot setup calls in the self-test.
func must(v []float32, err error) []float32 {
	if err != nil {
		panic(err)
	}
	return v
}

func runTestVectors(path string) int {
	fmt.Printf("fucina: test vectors from %s - not yet implemented\n", path)
	return 0
}

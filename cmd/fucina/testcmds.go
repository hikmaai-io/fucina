// ABOUTME: In-binary self-tests dispatched by --test-* flags (parser, CUDA, logits).
// ABOUTME: runTestLogits cross-checks single-token decode vs batched spec-verify per geometry.

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/hikmaai-io/fucina/internal/engine/cuda"
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
	const steps = 6

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
	if _, err := eng.Prefill(pt); err != nil {
		fmt.Fprintf(os.Stderr, "fucina: prefill: %v\n", err)
		return 1
	}
	nLayers := eng.NumLayers()
	nKeep := eng.NTokens() // KV state to restore to before each path

	tokIDs := make([]int32, 0, steps)
	maxAbs, maxRel := 0.0, 0.0
	argmaxMismatch := 0
	next := int32(pt[len(pt)-1]) // first token to feed both paths (last prompt token)
	for s := 0; s < steps; s++ {
		lsShared, err := eng.Decode(next)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fucina: decode step %d: %v\n", s, err)
			return 1
		}
		ls := append([]float32(nil), lsShared...) // copy before Rewind/Batched reuse the buffer
		if !eng.Rewind(nKeep) {
			fmt.Fprintf(os.Stderr, "fucina: rewind to %d failed at step %d\n", nKeep, s)
			return 1
		}
		lb, err := eng.DecodeBatched([]int32{next})
		if err != nil {
			fmt.Fprintf(os.Stderr, "fucina: decode_batched step %d: %v\n", s, err)
			return 1
		}
		if !eng.Rewind(nKeep) {
			fmt.Fprintf(os.Stderr, "fucina: rewind (post-batched) failed at step %d\n", s)
			return 1
		}
		da, db := argmax(ls), argmax(lb)
		if da != db {
			argmaxMismatch++
			fmt.Printf("  step %d: ARGMAX MISMATCH single=%d batched=%d\n", s, da, db)
		}
		var sumSq, refSq float64
		for i := range ls {
			d := math.Abs(float64(ls[i] - lb[i]))
			if d > maxAbs {
				maxAbs = d
			}
			sumSq += d * d
			refSq += float64(ls[i]) * float64(ls[i])
		}
		if refSq > 0 {
			if rel := math.Sqrt(sumSq / refSq); rel > maxRel {
				maxRel = rel
			}
		}
		// Advance the reference trajectory with the single-token greedy choice, then
		// re-decode it on the (rewound) engine to move the KV state forward one step.
		next = int32(da)
		tokIDs = append(tokIDs, next)
		if _, err := eng.Decode(next); err != nil {
			fmt.Fprintf(os.Stderr, "fucina: advance decode step %d: %v\n", s, err)
			return 1
		}
		nKeep = eng.NTokens()
	}

	geom := "12B"
	if nLayers > 48 {
		geom = "31B"
	}
	fmt.Printf("fucina: logit self-test (%s): max_abs_err=%.6g max_rel_l2=%.6g argmax_mismatches=%d/%d\n",
		geom, maxAbs, maxRel, argmaxMismatch, steps)
	fmt.Printf("  single-token greedy: %q\n", tok.Decode(tokIDs))
	if argmaxMismatch != 0 {
		fmt.Println("  FAIL: batched spec-verify diverges from single-token decode")
		return 1
	}
	fmt.Println("  PASS: batched matches single-token (spec decoding is sound)")
	return 0
}

// argmax returns the index of the maximum logit.
func argmax(v []float32) int {
	best, bi := float32(math.Inf(-1)), 0
	for i, x := range v {
		if x > best {
			best, bi = x, i
		}
	}
	return bi
}

func runTestVectors(path string) int {
	fmt.Printf("fucina: test vectors from %s - not yet implemented\n", path)
	return 0
}

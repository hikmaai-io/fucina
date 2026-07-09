package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/hikmaai-io/fucina/internal/engine/cuda"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

type jspaceTopToken struct {
	Token   string  `json:"token"`    // exact decoded vocabulary string (whitespace preserved)
	TokenID int32   `json:"token_id"` // retained only for unambiguous replay/steering
	Prob    float32 `json:"prob"`
}

type jspaceLayer struct {
	Layer int              `json:"layer"`
	Top   []jspaceTopToken `json:"top"`
}

type jspaceRecord struct {
	Type           string        `json:"type"`
	Timestamp      string        `json:"timestamp"`
	Turn           uint64        `json:"turn"`
	Step           int           `json:"step"`
	SourcePosition int           `json:"source_position"`
	SourceToken    string        `json:"source_token"`
	SourceTokenID  int32         `json:"source_token_id"`
	SampledToken   string        `json:"sampled_token"`
	SampledTokenID int32         `json:"sampled_token_id"`
	Layers         []jspaceLayer `json:"layers"`
}

type jspaceTracer struct {
	eng        *cuda.Engine
	tok        *tokenizer.Tokenizer
	file       *os.File
	enc        *json.Encoder
	path       string
	allowSteer bool
}

func newJSpaceTracer(eng *cuda.Engine, tok *tokenizer.Tokenizer, args CLIArgs) (*jspaceTracer, error) {
	if !args.JSpace {
		return nil, nil
	}
	f, err := os.OpenFile(args.JSpaceOut, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open J-space JSONL: %w", err)
	}
	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	return &jspaceTracer{eng: eng, tok: tok, file: f, enc: enc, path: args.JSpaceOut,
		allowSteer: args.JSpaceDebug}, nil
}

func (t *jspaceTracer) Close() error {
	if t == nil || t.file == nil {
		return nil
	}
	return t.file.Close()
}

// Record writes the J-space at the residual position that predicts sampledToken. Keeping the
// source and sampled tokens separate is essential: a causal LM's hidden state at "crook" predicts
// the following token, while workspace experiments interpret the state at "crook" itself.
func (t *jspaceTracer) Record(turn uint64, step, sourcePosition int,
	sourceToken, sampledToken int32) error {
	if t == nil {
		return nil
	}
	entries, err := t.eng.JSpaceSnapshot()
	if err != nil {
		return err
	}
	rec := jspaceRecord{
		Type: "token", Timestamp: time.Now().UTC().Format(time.RFC3339Nano), Turn: turn, Step: step,
		SourcePosition: sourcePosition, SourceToken: t.tok.Decode([]int32{sourceToken}),
		SourceTokenID: sourceToken, SampledToken: t.tok.Decode([]int32{sampledToken}),
		SampledTokenID: sampledToken, Layers: make([]jspaceLayer, len(entries)),
	}
	for i, entry := range entries {
		layer := jspaceLayer{Layer: entry.Layer, Top: make([]jspaceTopToken, len(entry.TokenIDs))}
		for k, id := range entry.TokenIDs {
			layer.Top[k] = jspaceTopToken{Token: t.tok.Decode([]int32{id}), TokenID: id, Prob: entry.Probs[k]}
		}
		rec.Layers[i] = layer
	}
	if err := t.enc.Encode(&rec); err != nil {
		return fmt.Errorf("write J-space JSONL: %w", err)
	}
	return nil
}

func parseJSpaceLayers(s string) ([]int, error) {
	if s == "" || strings.EqualFold(s, "all") {
		return nil, nil
	}
	var out []int
	for _, raw := range strings.Split(s, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(raw))
		if err != nil || v < 0 {
			return nil, fmt.Errorf("invalid layer %q", raw)
		}
		out = append(out, v)
	}
	return out, nil
}

// splitJSpaceCommand is a tiny shell-like splitter so exact decoded token strings can be supplied
// as words, including their significant leading whitespace: /jsteer " Paris" 0.15 all.
func splitJSpaceCommand(input string) ([]string, error) {
	var out []string
	var b strings.Builder
	var quote rune
	escaped := false
	started := false
	flush := func() {
		if started {
			out = append(out, b.String())
			b.Reset()
			started = false
		}
	}
	for _, r := range input {
		if escaped {
			b.WriteRune(r)
			started = true
			escaped = false
			continue
		}
		if r == '\\' && quote != 0 {
			escaped = true
			continue
		}
		if quote != 0 {
			if r == quote {
				quote = 0
			} else {
				b.WriteRune(r)
			}
			started = true
			continue
		}
		if r == '\'' || r == '"' {
			quote = r
			started = true
			continue
		}
		if r == ' ' || r == '\t' {
			flush()
			continue
		}
		b.WriteRune(r)
		started = true
	}
	if escaped || quote != 0 {
		return nil, fmt.Errorf("unterminated quoted token")
	}
	flush()
	return out, nil
}

// handleCommand handles J-space-only REPL commands. Quoted words are preferred; numeric ids are
// still accepted because distinct vocabulary entries can decode to the same visible string.
func (t *jspaceTracer) handleCommand(input string) (handled bool, err error) {
	if t == nil || !strings.HasPrefix(input, "/j") {
		return false, nil
	}
	parts, splitErr := splitJSpaceCommand(input)
	if splitErr != nil {
		return true, splitErr
	}
	if len(parts) == 0 {
		return false, nil
	}
	switch parts[0] {
	case "/jclear":
		if !t.allowSteer {
			return true, fmt.Errorf("steering requires --jspace-debug")
		}
		t.eng.ClearJSpaceSteer()
		fmt.Fprintln(os.Stderr, "fucina: J-space steering cleared")
		return true, nil
	case "/jdump":
		entries, snapErr := t.eng.JSpaceSnapshot()
		if snapErr != nil {
			return true, snapErr
		}
		fmt.Fprintf(os.Stderr, "fucina: latest J-space has %d fitted layers; automatic trace -> %s\n",
			len(entries), t.path)
		for _, entry := range entries {
			fmt.Fprintf(os.Stderr, "  L%-2d", entry.Layer)
			for k, id := range entry.TokenIDs {
				fmt.Fprintf(os.Stderr, " %q(%.3g)", t.tok.Decode([]int32{id}), entry.Probs[k])
			}
			fmt.Fprintln(os.Stderr)
		}
		return true, nil
	case "/jsteer":
		if !t.allowSteer {
			return true, fmt.Errorf("steering requires --jspace-debug")
		}
		if len(parts) < 3 {
			return true, fmt.Errorf("usage: /jsteer TOKEN_ID|TOKEN STRENGTH [LAYER,...|all]")
		}
		strength64, parseErr := strconv.ParseFloat(parts[2], 64)
		if parseErr != nil || strength64 < -1 || strength64 > 1 {
			return true, fmt.Errorf("strength must be in [-1,1]")
		}
		var tokenID int32
		if id, parseErr := strconv.ParseInt(parts[1], 10, 32); parseErr == nil {
			tokenID = int32(id)
		} else {
			ids := t.tok.Encode(parts[1], false, false)
			if len(ids) != 1 {
				return true, fmt.Errorf("%q encodes to %d tokens; use a numeric token id", parts[1], len(ids))
			}
			tokenID = ids[0]
		}
		layerArg := "all"
		if len(parts) > 3 {
			layerArg = parts[3]
		}
		layers, parseErr := parseJSpaceLayers(layerArg)
		if parseErr != nil {
			return true, parseErr
		}
		if setErr := t.eng.SetJSpaceSteer(tokenID, float32(strength64), layers); setErr != nil {
			return true, setErr
		}
		fmt.Fprintf(os.Stderr, "fucina: steering toward token %d %q strength=%g layers=%s\n",
			tokenID, t.tok.Decode([]int32{tokenID}), strength64, layerArg)
		return true, nil
	}
	return false, nil
}

// fucina-calibrate profiles sparse-MoE routing over a regenerable text/JSONL corpus.
// It emits a versioned importance sidecar consumed by future precision/residency policy tools.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/hikmaai-io/fucina/internal/engine/cuda"
	"github.com/hikmaai-io/fucina/internal/server/batch"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

type expertStat struct {
	Expert     int     `json:"expert"`
	Count      uint64  `json:"count"`
	Frequency  float64 `json:"frequency"`
	MeanWeight float64 `json:"mean_weight"`
	Importance float64 `json:"importance"`
}

type layerStat struct {
	Layer       int          `json:"layer"`
	Assignments uint64       `json:"assignments"`
	Experts     []expertStat `json:"experts"`
}

type activationStat struct {
	Elements   uint64  `json:"elements"`
	RMS        float64 `json:"rms"`
	MaxAbs     float64 `json:"max_abs"`
	Importance float64 `json:"importance"`
}

type sidecar struct {
	Format           string                    `json:"format"`
	CreatedUTC       string                    `json:"created_utc"`
	Model            string                    `json:"model"`
	Corpus           string                    `json:"corpus"`
	Documents        int                       `json:"documents"`
	Tokens           int                       `json:"tokens"`
	Layers           int                       `json:"layers"`
	ExpertCount      int                       `json:"expert_count"`
	TopK             int                       `json:"top_k"`
	HeatMap          []layerStat               `json:"expert_heat_map"`
	ActivationStats  map[string]activationStat `json:"activation_stats"`
	TensorImportance map[string]float64        `json:"tensor_importance"`
}

func corpusText(line []byte) string {
	s := strings.TrimSpace(string(line))
	if s == "" {
		return ""
	}
	var v any
	if json.Unmarshal(line, &v) != nil {
		return s
	}
	return extractText(v)
}

func extractText(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case []any:
		parts := make([]string, 0, len(x))
		for _, e := range x {
			if s := extractText(e); s != "" {
				parts = append(parts, s)
			}
		}
		return strings.Join(parts, "\n")
	case map[string]any:
		for _, k := range []string{"text", "prompt"} {
			if s, ok := x[k].(string); ok && s != "" {
				return s
			}
		}
		if msgs, ok := x["messages"]; ok {
			return extractText(msgs)
		}
		if c, ok := x["content"]; ok {
			return extractText(c)
		}
	}
	return ""
}

func buildSidecar(model, corpus string, docs, tokens int, p cuda.MoEProfile) sidecar {
	o := sidecar{Format: "fucina-imatrix-v1", CreatedUTC: time.Now().UTC().Format(time.RFC3339),
		Model: model, Corpus: corpus, Documents: docs, Tokens: tokens, Layers: p.Layers,
		ExpertCount: p.Experts, TopK: p.TopK, HeatMap: make([]layerStat, p.Layers),
		ActivationStats:  make(map[string]activationStat, p.Layers*5),
		TensorImportance: make(map[string]float64, p.Layers*p.Experts*3)}
	stageNames := []string{"mixer_input", "mixer_output_input", "moe_input", "expert_down_input", "shared_down_input"}
	maxRMS, maxAbs := 0.0, 0.0
	for l := 0; l < p.Layers; l++ {
		for s, name := range stageNames {
			i := l*len(stageNames) + s
			if i >= len(p.ActivationElements) {
				continue
			}
			n := p.ActivationElements[i]
			rms := 0.0
			if n > 0 {
				rms = math.Sqrt(p.ActivationSumSq[i] / float64(n))
			}
			a := activationStat{Elements: n, RMS: rms, MaxAbs: float64(p.ActivationMaxAbs[i])}
			o.ActivationStats[fmt.Sprintf("layers.%d.%s", l, name)] = a
			if rms > maxRMS {
				maxRMS = rms
			}
			if a.MaxAbs > maxAbs {
				maxAbs = a.MaxAbs
			}
		}
	}
	for k, a := range o.ActivationStats {
		r, m := 0.0, 0.0
		if maxRMS > 0 {
			r = a.RMS / maxRMS
		}
		if maxAbs > 0 {
			m = a.MaxAbs / maxAbs
		}
		a.Importance = 0.8*r + 0.2*m
		o.ActivationStats[k] = a
	}
	act := func(l, s int) float64 {
		return o.ActivationStats[fmt.Sprintf("layers.%d.%s", l, stageNames[s])].Importance
	}
	for l := 0; l < p.Layers; l++ {
		base := l * p.Experts
		var total uint64
		var maxFreq, maxMean float64
		for e := 0; e < p.Experts; e++ {
			total += p.Counts[base+e]
		}
		stats := make([]expertStat, p.Experts)
		for e := range stats {
			c := p.Counts[base+e]
			freq := 0.0
			if total > 0 {
				freq = float64(c) / float64(total)
			}
			mean := 0.0
			if c > 0 {
				mean = p.WeightSums[base+e] / float64(c)
			}
			stats[e] = expertStat{Expert: e, Count: c, Frequency: freq, MeanWeight: mean}
			if freq > maxFreq {
				maxFreq = freq
			}
			if mean > maxMean {
				maxMean = mean
			}
		}
		for i := range stats {
			f, w := 0.0, 0.0
			if maxFreq > 0 {
				f = stats[i].Frequency / maxFreq
			}
			if maxMean > 0 {
				w = stats[i].MeanWeight / maxMean
			}
			stats[i].Importance = 0.7*f + 0.3*w
			e := stats[i].Expert
			for _, proj := range []string{"gate_proj", "up_proj"} {
				o.TensorImportance[fmt.Sprintf("layers.%d.mlp.experts.%d.%s.weight", l, e, proj)] = stats[i].Importance * (0.5 + 0.5*act(l, 2))
			}
			o.TensorImportance[fmt.Sprintf("layers.%d.mlp.experts.%d.down_proj.weight", l, e)] = stats[i].Importance * (0.5 + 0.5*act(l, 3))
		}
		sort.Slice(stats, func(i, j int) bool { return stats[i].Count > stats[j].Count })
		o.HeatMap[l] = layerStat{Layer: l, Assignments: total, Experts: stats}
		for _, proj := range []string{"gate_proj", "up_proj"} {
			o.TensorImportance[fmt.Sprintf("layers.%d.mlp.shared_expert.%s.weight", l, proj)] = math.Max(0.9, act(l, 2))
		}
		o.TensorImportance[fmt.Sprintf("layers.%d.mlp.shared_expert.down_proj.weight", l)] = math.Max(0.9, act(l, 4))
		o.TensorImportance[fmt.Sprintf("layers.%d.mlp.gate.weight", l)] = math.Max(0.9, act(l, 2))
		if (l+1)%4 == 0 {
			for _, pname := range []string{"q_proj", "k_proj", "v_proj"} {
				o.TensorImportance[fmt.Sprintf("layers.%d.self_attn.%s.weight", l, pname)] = act(l, 0)
			}
			o.TensorImportance[fmt.Sprintf("layers.%d.self_attn.o_proj.weight", l)] = act(l, 1)
		} else {
			for _, pname := range []string{"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"} {
				o.TensorImportance[fmt.Sprintf("layers.%d.linear_attn.%s.weight", l, pname)] = act(l, 0)
			}
			o.TensorImportance[fmt.Sprintf("layers.%d.linear_attn.out_proj.weight", l)] = act(l, 1)
		}
		o.TensorImportance[fmt.Sprintf("layers.%d.input_layernorm.weight", l)] = 1
		o.TensorImportance[fmt.Sprintf("layers.%d.post_attention_layernorm.weight", l)] = 1
	}
	return o
}

func loadTokenizer(model, override string) (*tokenizer.Tokenizer, error) {
	path := override
	if path == "" {
		dir := model
		if fi, err := os.Stat(model); err != nil || !fi.IsDir() {
			dir = filepath.Dir(model)
		}
		path = filepath.Join(dir, "tokenizer.json")
	}
	if strings.HasSuffix(path, ".json") {
		return tokenizer.NewFromHFJSON(path)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return tokenizer.New(b, int64(len(b)))
}

func main() {
	runtime.LockOSThread()
	model := flag.String("m", "", "sparse Qwen safetensors/GGUF checkpoint")
	corpus := flag.String("corpus", "", "plain-text or JSONL calibration corpus")
	out := flag.String("out", "fucina.imatrix.json", "output importance sidecar")
	tokPath := flag.String("tokenizer", "", "tokenizer.json or tokenizer GGUF override")
	maxTokens := flag.Int("max-tokens", 3_000_000, "maximum corpus tokens")
	ctx := flag.Int("ctx", 8192, "maximum tokens per corpus record")
	flag.Parse()
	if *model == "" || *corpus == "" {
		flag.Usage()
		os.Exit(2)
	}
	if *maxTokens < 1 || *ctx < 1 {
		log.Fatal("max-tokens and ctx must be positive")
	}

	os.Setenv("FUCINA_PAGED_KV", "1")
	os.Setenv("FUCINA_PAGED_MAXSEQS", "1")
	tok, err := loadTokenizer(*model, *tokPath)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}
	eng, err := cuda.NewEngine(cuda.Config{ModelPath: *model, ContextSize: uint32(*ctx), GPUMemUtil: 0.90})
	if err != nil {
		log.Fatal(err)
	}
	defer eng.Close()
	eng.SetPrefixCache(false)
	if err = eng.StartMoEProfile(); err != nil {
		log.Fatal(err)
	}

	f, err := os.Open(*corpus)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 16*1024*1024)
	docs, ntok := 0, 0
	params := batch.SeqParams{Temperature: 0}
	for sc.Scan() && ntok < *maxTokens {
		text := corpusText(sc.Bytes())
		if text == "" {
			continue
		}
		ids := tok.Encode(text, false, false)
		if len(ids) == 0 {
			continue
		}
		remain := *maxTokens - ntok
		if len(ids) > remain {
			ids = ids[:remain]
		}
		if len(ids) > *ctx {
			ids = ids[:*ctx]
		}
		slot, _, e := eng.SeqAdd(ids, params)
		if e != nil {
			log.Fatalf("record %d: %v", docs+1, e)
		}
		eng.SeqRemove(slot)
		docs++
		ntok += len(ids)
		if docs%100 == 0 {
			log.Printf("profiled %d documents / %d tokens", docs, ntok)
		}
	}
	if err = sc.Err(); err != nil {
		log.Fatal(err)
	}
	p, err := eng.MoEProfile()
	if err != nil {
		log.Fatal(err)
	}
	result := buildSidecar(*model, *corpus, docs, ntok, p)
	b, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	b = append(b, '\n')
	if err = os.WriteFile(*out, b, 0644); err != nil {
		log.Fatal(err)
	}
	log.Printf("wrote %s (%d documents, %d tokens, %d layers × %d experts)", *out, docs, ntok, p.Layers, p.Experts)
}

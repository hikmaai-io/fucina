// fucina-calibrate profiles sparse-MoE routing over a regenerable text/JSONL corpus.
// It emits a versioned importance sidecar consumed by future precision/residency policy tools.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
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

type sidecar struct {
	Format           string             `json:"format"`
	CreatedUTC       string             `json:"created_utc"`
	Model            string             `json:"model"`
	Corpus           string             `json:"corpus"`
	Documents        int                `json:"documents"`
	Tokens           int                `json:"tokens"`
	Layers           int                `json:"layers"`
	ExpertCount      int                `json:"expert_count"`
	TopK             int                `json:"top_k"`
	HeatMap          []layerStat        `json:"expert_heat_map"`
	TensorImportance map[string]float64 `json:"tensor_importance"`
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
		TensorImportance: make(map[string]float64, p.Layers*p.Experts*3)}
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
			for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
				o.TensorImportance[fmt.Sprintf("layers.%d.mlp.experts.%d.%s.weight", l, e, proj)] = stats[i].Importance
			}
		}
		sort.Slice(stats, func(i, j int) bool { return stats[i].Count > stats[j].Count })
		o.HeatMap[l] = layerStat{Layer: l, Assignments: total, Experts: stats}
		for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
			o.TensorImportance[fmt.Sprintf("layers.%d.mlp.shared_expert.%s.weight", l, proj)] = 1
		}
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

package main

import (
	"flag"
	"testing"

	gemserver "github.com/hikmaai-io/fucina/internal/server"
)

// newTestFlagSet returns a fresh FlagSet that does NOT call os.Exit on error,
// so parseArgs can be exercised in isolation.
func newTestFlagSet() *flag.FlagSet {
	return flag.NewFlagSet("fucina-test", flag.ContinueOnError)
}

func mustParse(t *testing.T, argv []string) (CLIArgs, testFlags) {
	t.Helper()
	a, tf, err := parseArgs(newTestFlagSet(), argv)
	if err != nil {
		t.Fatalf("parseArgs(%v) returned error: %v", argv, err)
	}
	return a, tf
}

func TestParseArgsDefaults(t *testing.T) {
	a, tf := mustParse(t, nil)

	if a.ContextSize != 262144 {
		t.Errorf("ContextSize = %d, want 262144", a.ContextSize)
	}
	if a.Temperature != 1.0 {
		t.Errorf("Temperature = %v, want 1.0", a.Temperature)
	}
	if a.TopK != 64 {
		t.Errorf("TopK = %d, want 64", a.TopK)
	}
	if a.TopP != 0.95 {
		t.Errorf("TopP = %v, want 0.95", a.TopP)
	}
	if a.Predict != 512 {
		t.Errorf("Predict = %d, want 512", a.Predict)
	}
	if a.Host != "127.0.0.1" {
		t.Errorf("Host = %q, want 127.0.0.1", a.Host)
	}
	if a.Port != 8080 {
		t.Errorf("Port = %d, want 8080", a.Port)
	}
	if a.Thinking != "off" {
		t.Errorf("Thinking = %q, want off", a.Thinking)
	}
	if gemserver.ParseThinkingLevel(a.Thinking) != false {
		t.Errorf("default thinking parses to true, want false")
	}
	if a.Spec != true {
		t.Errorf("Spec = %v, want true", a.Spec)
	}
	if a.DraftK != 6 {
		t.Errorf("DraftK = %d, want 6", a.DraftK)
	}
	if a.Seed != -1 {
		t.Errorf("Seed = %d, want -1", a.Seed)
	}
	if a.GPUMemUtil != 0.90 {
		t.Errorf("GPUMemUtil = %v, want 0.90", a.GPUMemUtil)
	}

	// --gpu-mem-util overrides the default.
	if b, _ := mustParse(t, []string{"--gpu-mem-util", "0.20"}); b.GPUMemUtil != 0.20 {
		t.Errorf("GPUMemUtil = %v, want 0.20", b.GPUMemUtil)
	}

	// No test-flag dispatch by default.
	if tf.parser || tf.cuda || tf.vectors != "" {
		t.Errorf("test flags should be unset by default, got %+v", tf)
	}
}

func TestParseArgsShortLongAliases(t *testing.T) {
	cases := []struct {
		name  string
		short []string
		long  []string
		check func(CLIArgs) (got, want any)
	}{
		{
			name:  "model",
			short: []string{"-m", "x.gguf"},
			long:  []string{"--model", "x.gguf"},
			check: func(a CLIArgs) (any, any) { return a.ModelPath, "x.gguf" },
		},
		{
			name:  "prompt",
			short: []string{"-p", "hi"},
			long:  []string{"--prompt", "hi"},
			check: func(a CLIArgs) (any, any) { return a.Prompt, "hi" },
		},
		{
			name:  "ctx",
			short: []string{"-c", "2048"},
			long:  []string{"--ctx", "2048"},
			check: func(a CLIArgs) (any, any) { return a.ContextSize, 2048 },
		},
		{
			name:  "predict",
			short: []string{"-n", "100"},
			long:  []string{"--predict", "100"},
			check: func(a CLIArgs) (any, any) { return a.Predict, 100 },
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			aShort, _ := mustParse(t, tc.short)
			aLong, _ := mustParse(t, tc.long)

			gotShort, want := tc.check(aShort)
			gotLong, _ := tc.check(aLong)

			if gotShort != want {
				t.Errorf("short %v: got %v, want %v", tc.short, gotShort, want)
			}
			if gotLong != want {
				t.Errorf("long %v: got %v, want %v", tc.long, gotLong, want)
			}
			if gotShort != gotLong {
				t.Errorf("short/long mismatch: %v vs %v", gotShort, gotLong)
			}
		})
	}
}

func TestParseArgsThinkingPassthrough(t *testing.T) {
	for _, level := range []string{"off", "on", "low", "mid", "high", "xhigh"} {
		a, _ := mustParse(t, []string{"--thinking", level})
		if a.Thinking != level {
			t.Errorf("--thinking %s: Thinking = %q, want %q", level, a.Thinking, level)
		}
	}
}

func TestParseArgsSeedDefault(t *testing.T) {
	a, _ := mustParse(t, nil)
	if a.Seed != -1 {
		t.Errorf("Seed default = %d, want -1", a.Seed)
	}
	a, _ = mustParse(t, []string{"--seed", "42"})
	if a.Seed != 42 {
		t.Errorf("Seed = %d, want 42", a.Seed)
	}
}

func TestParseArgsTestFlags(t *testing.T) {
	_, tf := mustParse(t, []string{"--test-parser"})
	if !tf.parser {
		t.Error("--test-parser did not set tf.parser")
	}
	_, tf = mustParse(t, []string{"--test-cuda"})
	if !tf.cuda {
		t.Error("--test-cuda did not set tf.cuda")
	}
	_, tf = mustParse(t, []string{"--test-vectors", "vec.json"})
	if tf.vectors != "vec.json" {
		t.Errorf("--test-vectors = %q, want vec.json", tf.vectors)
	}
}

func TestParseArgsJSpaceDebugImpliesTraceAndInteractive(t *testing.T) {
	a, _ := mustParse(t, []string{"--jspace-debug", "--jlens", "lens.fjls",
		"--jspace-out", "trace.jsonl", "--jspace-top-k", "12"})
	if !a.JSpace || !a.JSpaceDebug || !a.Interactive {
		t.Fatalf("JSpace=%v JSpaceDebug=%v Interactive=%v, want all true",
			a.JSpace, a.JSpaceDebug, a.Interactive)
	}
	if a.JLENSPath != "lens.fjls" || a.JSpaceOut != "trace.jsonl" || a.JSpaceTopK != 12 {
		t.Fatalf("unexpected J-space args: %+v", a)
	}
}

func TestParseArgsBatchImpliesPagedKV(t *testing.T) {
	// Default: both off.
	a, _ := mustParse(t, nil)
	if a.Batch || a.PagedKV {
		t.Errorf("defaults: Batch=%v PagedKV=%v, want both false", a.Batch, a.PagedKV)
	}
	// --paged-kv alone enables the paged engine but not the scheduler.
	a, _ = mustParse(t, []string{"--paged-kv"})
	if !a.PagedKV || a.Batch {
		t.Errorf("--paged-kv: PagedKV=%v Batch=%v, want true/false", a.PagedKV, a.Batch)
	}
	// --batch implies --paged-kv (scheduler is a no-op without the paged engine).
	a, _ = mustParse(t, []string{"--batch"})
	if !a.Batch || !a.PagedKV {
		t.Errorf("--batch: Batch=%v PagedKV=%v, want both true", a.Batch, a.PagedKV)
	}
}

func TestParseThinkingLevel(t *testing.T) {
	falseInputs := []string{"off", "none", "false"}
	for _, in := range falseInputs {
		if gemserver.ParseThinkingLevel(in) != false {
			t.Errorf("ParseThinkingLevel(%q) = true, want false", in)
		}
	}
	trueInputs := []string{"on", "low", "mid", "high", "xhigh", "true"}
	for _, in := range trueInputs {
		if gemserver.ParseThinkingLevel(in) != true {
			t.Errorf("ParseThinkingLevel(%q) = false, want true", in)
		}
	}
	if gemserver.ParseThinkingLevel("banana") != false {
		t.Error("ParseThinkingLevel(unknown) = true, want false")
	}
}

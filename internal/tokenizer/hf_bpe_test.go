package tokenizer

import (
	"os"
	"reflect"
	"testing"
)

// hfTokenizerPath returns the tokenizer.json to test against (env override, else the dev
// checkpoint), or "" to skip — the 32 MB file is not a committed fixture.
func hfTokenizerPath() string {
	if p := os.Getenv("FUCINA_HF_TOKENIZER"); p != "" {
		return p
	}
	const dev = "/home/mauromedda/hack/gem4d/gemma-4-12B-it-NVFP4/tokenizer.json"
	if _, err := os.Stat(dev); err == nil {
		return dev
	}
	return ""
}

// Ground truth captured from the HF `tokenizers` library on the real Gemma-4 tokenizer.json
// (encode with add_special_tokens=False). The Go BPE must reproduce these exactly.
var bpeGroundTruth = []struct {
	text string
	ids  []int32
}{
	{"Hello, world!", []int32{9259, 236764, 1902, 236888}},
	{"The capital of France is Paris.", []int32{818, 5279, 529, 7001, 563, 9079, 236761}},
	{"  leading and  double  spaces ", []int32{138, 26016, 532, 138, 7902, 138, 35220, 236743}},
	{"newlines\nand\ttabs", []int32{208697, 107, 624, 255968, 39218}},
	{"Photosynthesis is the process by which green plants", []int32{49660, 30445, 563, 506, 1657, 684, 837, 3826, 6485}},
	{"unicode: café — naïve — Ωμέγα", []int32{70926, 236787, 33443, 2192, 120362, 2192, 76858, 20961, 103132}},
	{"emoji 🚀 and 🔥 bytes", []int32{67906, 236743, 242015, 532, 128728, 17234}},
	{"code: def f(x): return x*2", []int32{3970, 236787, 1096, 517, 236769, 236781, 1473, 994, 1123, 236829, 236778}},
	{"<bos>hi<eos>", []int32{2, 2202, 1}},
	{"", []int32{}},
	{"a", []int32{236746}},
	{" ", []int32{236743}},
}

func TestHFBPEEncodeMatchesGroundTruth(t *testing.T) {
	path := hfTokenizerPath()
	if path == "" {
		t.Skip("no tokenizer.json available (set FUCINA_HF_TOKENIZER)")
	}
	tk, err := NewFromHFJSON(path)
	if err != nil {
		t.Fatalf("NewFromHFJSON: %v", err)
	}
	if !tk.bpe {
		t.Fatal("expected bpe mode")
	}
	if tk.BOS != 2 || tk.EOS != 1 {
		t.Fatalf("control ids: BOS=%d EOS=%d (want 2,1)", tk.BOS, tk.EOS)
	}
	for _, c := range bpeGroundTruth {
		got := tk.Encode(c.text, false, false)
		if len(got) == 0 && len(c.ids) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, c.ids) {
			t.Errorf("Encode(%q):\n  got  %v\n  want %v", c.text, got, c.ids)
		}
	}
}

func TestHFBPERoundTrip(t *testing.T) {
	path := hfTokenizerPath()
	if path == "" {
		t.Skip("no tokenizer.json available")
	}
	tk, err := NewFromHFJSON(path)
	if err != nil {
		t.Fatalf("NewFromHFJSON: %v", err)
	}
	// Decode(Encode(x)) reproduces x for marker-free text (Decode maps ▁→space, reassembles bytes).
	for _, s := range []string{
		"Hello, world!", "The quick brown fox.", "café naïve", "emoji 🚀 fire 🔥", "tabs\tand\nnewlines",
	} {
		if out := tk.Decode(tk.Encode(s, false, false)); out != s {
			t.Errorf("round-trip mismatch:\n  in  %q\n  out %q", s, out)
		}
	}
}

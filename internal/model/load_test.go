package model

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func TestLoadDescriptorDispatchesGGUFBuilder(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gemma.gguf")
	writeManifestGGUF(t, path, gemmaGGUF("Q4_0", "Q6_K"))

	d, err := LoadDescriptor(path)
	if err != nil {
		t.Fatal(err)
	}
	if d.Family() != "gemma-4" || d.SourceFormat() != FormatGGUF || d.SourceQuantization() != "Q4_0-QAT" {
		t.Fatalf("descriptor=%+v", d.Snapshot())
	}
	if d.Geometry().Layers != 48 || len(d.Mixers()) != 48 {
		t.Fatalf("geometry=%+v mixers=%d", d.Geometry(), len(d.Mixers()))
	}
	first := d.Fingerprints()
	d2, err := LoadDescriptor(path)
	if err != nil {
		t.Fatal(err)
	}
	if first != d2.Fingerprints() {
		t.Fatalf("fingerprint changed: first=%+v second=%+v", first, d2.Fingerprints())
	}
}

func TestFromGGUFDerivesVocabFromTokenArray(t *testing.T) {
	metadata := gemmaGGUF("Q8_0", "Q8_0")
	delete(metadata.Values, "gemma4.vocab_size")
	delete(metadata.Values, "tokenizer.ggml.token_count")
	metadata.Values["tokenizer.ggml.tokens"] = make([]any, 262144)
	d, err := FromGGUF(metadata)
	if err != nil {
		t.Fatal(err)
	}
	if d.Geometry().VocabSize != 262144 {
		t.Fatalf("vocab=%d", d.Geometry().VocabSize)
	}
}

func TestLoadDescriptorDispatchesConfigBuilder(t *testing.T) {
	d, err := LoadDescriptor(fixtureGemmaNVFP4(t))
	if err != nil {
		t.Fatal(err)
	}
	if d.SourceFormat() != FormatSafetensors || d.Architecture() != "gemma4" {
		t.Fatalf("descriptor=%+v", d.Snapshot())
	}
}

func TestReadGGUFMetadataRejectsCountBomb(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bomb.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range []any{uint32(ggufMagic), uint32(3), uint64(0), uint64(maxGGUFKVCount + 1)} {
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := readGGUFMetadata(path); err == nil || !strings.Contains(err.Error(), "metadata count") {
		t.Fatalf("error=%v", err)
	}
}

func writeManifestGGUF(t *testing.T, path string, metadata GGUFMetadata) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	write := func(v any) {
		t.Helper()
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	writeString := func(s string) {
		write(uint64(len(s)))
		if _, err := f.WriteString(s); err != nil {
			t.Fatal(err)
		}
	}

	write(uint32(ggufMagic))
	write(uint32(3))
	write(uint64(len(metadata.Tensors)))
	write(uint64(len(metadata.Values)))

	keys := make([]string, 0, len(metadata.Values))
	for key := range metadata.Values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		writeString(key)
		writeTestGGUFValue(t, f, metadata.Values[key])
	}

	names := make([]string, 0, len(metadata.Tensors))
	for name := range metadata.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		tensor := metadata.Tensors[name]
		writeString(name)
		write(uint32(len(tensor.Shape)))
		for _, dim := range tensor.Shape {
			write(uint64(dim))
		}
		write(testGGMLType(t, tensor.Encoding))
		write(uint64(0))
	}
}

func writeTestGGUFValue(t *testing.T, f *os.File, value any) {
	t.Helper()
	write := func(v any) {
		t.Helper()
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	writeString := func(s string) {
		write(uint64(len(s)))
		if _, err := f.WriteString(s); err != nil {
			t.Fatal(err)
		}
	}
	switch v := value.(type) {
	case string:
		write(ggufString)
		writeString(v)
	case int:
		write(ggufInt32)
		write(int32(v))
	case float64:
		write(ggufFloat64)
		write(math.Float64bits(v))
	case bool:
		write(ggufBool)
		if v {
			write(uint8(1))
		} else {
			write(uint8(0))
		}
	case []int:
		write(ggufArray)
		write(ggufInt32)
		write(uint64(len(v)))
		for _, n := range v {
			write(int32(n))
		}
	case []bool:
		write(ggufArray)
		write(ggufBool)
		write(uint64(len(v)))
		for _, b := range v {
			if b {
				write(uint8(1))
			} else {
				write(uint8(0))
			}
		}
	default:
		t.Fatalf("unsupported test GGUF value %T", value)
	}
}

func testGGMLType(t *testing.T, encoding string) uint32 {
	t.Helper()
	for typ := uint32(0); typ <= 30; typ++ {
		if ggmlEncoding(typ) == encoding {
			return typ
		}
	}
	t.Fatalf("unsupported test encoding %q", encoding)
	return 0
}

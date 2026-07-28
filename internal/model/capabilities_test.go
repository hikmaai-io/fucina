package model

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

type capabilityDocument struct {
	Schema  int
	Claims  map[string]string
	Formats map[string]string
}

func TestCapabilitiesYAMLReconcilesWithManifestBuilders(t *testing.T) {
	path := filepath.Join("..", "..", "docs", "capabilities.yaml")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	doc, err := parseCapabilityYAML(b)
	if err != nil {
		t.Fatal(err)
	}
	if err := reconcileCapabilities(t, doc); err != nil {
		t.Fatal(err)
	}
}

func TestCapabilitiesYAMLContradictionIsDetected(t *testing.T) {
	path := filepath.Join("..", "..", "docs", "capabilities.yaml")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	contradiction := strings.Replace(string(b), "batch_mtp_gemma4: true", "batch_mtp_gemma4: false", 1)
	doc, err := parseCapabilityYAML([]byte(contradiction))
	if err != nil {
		t.Fatal(err)
	}
	if err := reconcileCapabilities(t, doc); err == nil {
		t.Fatal("contradictory capability claim unexpectedly reconciled")
	}
}

func reconcileCapabilities(t *testing.T, doc capabilityDocument) error {
	t.Helper()
	if doc.Schema != 1 {
		return fmt.Errorf("capabilities schema=%d, want 1", doc.Schema)
	}
	q4, err := FromGGUF(gemmaGGUF("Q4_0", "Q6_K"))
	if err != nil {
		return err
	}
	q8, err := FromGGUF(gemmaGGUF("Q8_0", "Q8_0"))
	if err != nil {
		return err
	}
	gemNV, err := FromConfigJSON(fixtureGemmaNVFP4(t))
	if err != nil {
		return err
	}
	q9, err := FromConfigJSON(fixtureQwenFP8(t, "Qwen3.5-9B-FP8", 4096, 12288, 32, 16, 4, 32, 0))
	if err != nil {
		return err
	}
	q27, err := FromConfigJSON(fixtureQwenFP8(t, "Qwen3.6-27B-FP8", 5120, 17408, 64, 24, 4, 48, 0))
	if err != nil {
		return err
	}
	q35, err := FromConfigJSON(fixtureQwenFP8(t, "Qwen3.5-35B-A3B-FP8", 2048, 512, 40, 16, 2, 32, 256))
	if err != nil {
		return err
	}
	e4b, err := FromConfigJSON(fixtureE4B(t))
	if err != nil {
		return err
	}
	models := []*ModelDescriptor{q4, q8, gemNV, q9, q27, q35, e4b}
	actualClaims := map[string]bool{
		"batch_mtp_gemma4":            q4.Capabilities().BatchMTP,
		"continuous_batching_default": allModels(models, func(c ExecutionCapabilities) bool { return c.ContinuousBatchingDefault }),
		"e4b_serving":                 e4b.Capabilities().E4BServing,
		"legacy_qwen3":                anyModel(models, func(c ExecutionCapabilities) bool { return c.LegacyQwen3 }),
	}
	if len(doc.Claims) != len(actualClaims) {
		return fmt.Errorf("capability claim key count=%d, source=%d", len(doc.Claims), len(actualClaims))
	}
	for key, want := range actualClaims {
		raw, ok := doc.Claims[key]
		if !ok {
			return fmt.Errorf("capability claim %q missing", key)
		}
		got, err := strconv.ParseBool(raw)
		if err != nil {
			return fmt.Errorf("capability claim %q: %w", key, err)
		}
		if got != want {
			return fmt.Errorf("capability claim %q=%v contradicts source-derived %v", key, got, want)
		}
	}
	actualFormats := map[string]Qualification{
		"gemma4_12b_gguf_q4_0_qat":       q4.Qualification(),
		"gemma4_12b_gguf_q8_0":           q8.Qualification(),
		"gemma4_12b_safetensors_nvfp4":   gemNV.Qualification(),
		"qwen35_9b_safetensors_fp8":      q9.Qualification(),
		"qwen36_27b_safetensors_fp8":     q27.Qualification(),
		"qwen35_35b_a3b_safetensors_fp8": q35.Qualification(),
		"gemma4_e4b_safetensors_bf16":    e4b.Qualification(),
	}
	if len(doc.Formats) != len(actualFormats) {
		return fmt.Errorf("format claim key count=%d, source=%d", len(doc.Formats), len(actualFormats))
	}
	for key, want := range actualFormats {
		got, ok := doc.Formats[key]
		if !ok {
			return fmt.Errorf("format claim %q missing", key)
		}
		if Qualification(got) != want {
			return fmt.Errorf("format claim %q=%s contradicts builder %s", key, got, want)
		}
	}
	return nil
}

func allModels(models []*ModelDescriptor, p func(ExecutionCapabilities) bool) bool {
	for _, m := range models {
		if !p(m.Capabilities()) {
			return false
		}
	}
	return true
}
func anyModel(models []*ModelDescriptor, p func(ExecutionCapabilities) bool) bool {
	for _, m := range models {
		if p(m.Capabilities()) {
			return true
		}
	}
	return false
}

// parseCapabilityYAML intentionally accepts the small, strict mapping-only schema above. Avoiding
// a broad YAML dependency keeps the production module dependency-free and makes unknown nesting,
// duplicate keys, and malformed claims fail closed in CI.
func parseCapabilityYAML(data []byte) (capabilityDocument, error) {
	doc := capabilityDocument{Claims: map[string]string{}, Formats: map[string]string{}}
	section := ""
	for lineNo, line := range strings.Split(string(data), "\n") {
		line = strings.TrimRight(line, " \t\r")
		trim := strings.TrimSpace(line)
		if trim == "" || strings.HasPrefix(trim, "#") {
			continue
		}
		indent := len(line) - len(strings.TrimLeft(line, " "))
		if strings.Contains(line[:indent], "\t") {
			return doc, fmt.Errorf("line %d: tabs forbidden", lineNo+1)
		}
		parts := strings.SplitN(trim, ":", 2)
		if len(parts) != 2 {
			return doc, fmt.Errorf("line %d: expected key: value", lineNo+1)
		}
		key, val := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
		if indent == 0 {
			section = ""
			switch key {
			case "schema":
				n, err := strconv.Atoi(val)
				if err != nil {
					return doc, fmt.Errorf("line %d: %w", lineNo+1, err)
				}
				doc.Schema = n
			case "claims", "formats":
				if val != "" {
					return doc, fmt.Errorf("line %d: section must not have scalar", lineNo+1)
				}
				section = key
			default:
				return doc, fmt.Errorf("line %d: unknown top-level key %q", lineNo+1, key)
			}
			continue
		}
		if indent != 2 || section == "" {
			return doc, fmt.Errorf("line %d: unsupported indentation", lineNo+1)
		}
		target := doc.Claims
		if section == "formats" {
			target = doc.Formats
		}
		if _, dup := target[key]; dup {
			return doc, fmt.Errorf("line %d: duplicate %q", lineNo+1, key)
		}
		if val == "" {
			return doc, fmt.Errorf("line %d: empty value", lineNo+1)
		}
		target[key] = val
	}
	return doc, nil
}

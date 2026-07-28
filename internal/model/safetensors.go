package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const maxSafetensorsHeader = 256 << 20

type safetensorHeader struct {
	DType       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

func readCheckpointTensorIndex(dir string) (map[string]TensorInfo, error) {
	files, err := checkpointShards(dir)
	if err != nil {
		return nil, err
	}
	out := make(map[string]TensorInfo)
	for _, file := range files {
		part, err := readSafetensorsHeader(filepath.Join(dir, file))
		if err != nil {
			return nil, fmt.Errorf("tensor index %s: %w", file, err)
		}
		for name, t := range part {
			if _, exists := out[name]; exists {
				return nil, fmt.Errorf("tensor index: duplicate tensor %q", name)
			}
			out[name] = t
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("tensor index: checkpoint contains no tensors")
	}
	return out, nil
}

func checkpointShards(dir string) ([]string, error) {
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	if b, err := os.ReadFile(indexPath); err == nil {
		var idx struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(b, &idx); err != nil {
			return nil, fmt.Errorf("tensor index: parse %s: %w", indexPath, err)
		}
		set := map[string]struct{}{}
		for _, f := range idx.WeightMap {
			if filepath.Base(f) != f {
				return nil, fmt.Errorf("tensor index: shard path %q is not a basename", f)
			}
			set[f] = struct{}{}
		}
		files := make([]string, 0, len(set))
		for f := range set {
			files = append(files, f)
		}
		sort.Strings(files)
		if len(files) == 0 {
			return nil, fmt.Errorf("tensor index: empty weight_map")
		}
		return files, nil
	}
	matches, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return nil, err
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("tensor index: no model.safetensors.index.json or *.safetensors in %s", dir)
	}
	files := make([]string, len(matches))
	for i, p := range matches {
		files[i] = filepath.Base(p)
	}
	sort.Strings(files)
	return files, nil
}

func readSafetensorsHeader(path string) (map[string]TensorInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	var prefix [8]byte
	if _, err := io.ReadFull(f, prefix[:]); err != nil {
		return nil, fmt.Errorf("short header length: %w", err)
	}
	n := binary.LittleEndian.Uint64(prefix[:])
	if n == 0 || n > maxSafetensorsHeader {
		return nil, fmt.Errorf("invalid header length %d", n)
	}
	b := make([]byte, int(n))
	if _, err := io.ReadFull(f, b); err != nil {
		return nil, fmt.Errorf("short JSON header: %w", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("invalid JSON header: %w", err)
	}
	out := make(map[string]TensorInfo, len(raw))
	type interval struct {
		name       string
		begin, end int64
	}
	intervals := make([]interval, 0, len(raw))
	var mismatches []string
	dataBytes := stat.Size() - 8 - int64(n)
	for name, msg := range raw {
		if name == "__metadata__" {
			continue
		}
		var h safetensorHeader
		if err := json.Unmarshal(msg, &h); err != nil {
			mismatches = append(mismatches, fmt.Sprintf("%s: %v", name, err))
			continue
		}
		if name == "" || strings.ContainsRune(name, '\x00') {
			mismatches = append(mismatches, "invalid tensor name")
			continue
		}
		elements := int64(1)
		shapeOK := true
		for _, d := range h.Shape {
			if d <= 0 || elements > int64(^uint64(0)>>1)/d {
				mismatches = append(mismatches, fmt.Sprintf("%s: invalid or overflowing shape", name))
				shapeOK = false
				break
			}
			elements *= d
		}
		if len(h.DataOffsets) != 2 || h.DataOffsets[0] < 0 || h.DataOffsets[1] < h.DataOffsets[0] {
			mismatches = append(mismatches, fmt.Sprintf("%s: invalid data_offsets", name))
			continue
		}
		span := h.DataOffsets[1] - h.DataOffsets[0]
		width, known := safetensorsDTypeBytes(canonicalEncoding(h.DType))
		if shapeOK && known && (elements > int64(^uint64(0)>>1)/width || span != elements*width) {
			mismatches = append(mismatches, fmt.Sprintf("%s: data span %d does not match shape/dtype bytes", name, span))
		}
		if h.DataOffsets[1] > dataBytes {
			mismatches = append(mismatches, fmt.Sprintf("%s: data_offsets exceed file payload", name))
		}
		intervals = append(intervals, interval{name, h.DataOffsets[0], h.DataOffsets[1]})
		out[name] = TensorInfo{Shape: append([]int64(nil), h.Shape...), Encoding: canonicalEncoding(h.DType), Bytes: span}
	}
	sort.Slice(intervals, func(i, j int) bool { return intervals[i].begin < intervals[j].begin })
	for i := 1; i < len(intervals); i++ {
		if intervals[i].begin < intervals[i-1].end {
			mismatches = append(mismatches, fmt.Sprintf("%s: data range overlaps %s", intervals[i].name, intervals[i-1].name))
		}
	}
	if len(mismatches) > 0 {
		sort.Strings(mismatches)
		return nil, fmt.Errorf("malformed tensor entries: %s", strings.Join(mismatches, "; "))
	}
	return out, nil
}

func safetensorsDTypeBytes(dtype string) (int64, bool) {
	switch canonicalEncoding(dtype) {
	case "BOOL", "U8", "I8", "F8_E4M3", "F8_E4M3FN", "F8_E5M2":
		return 1, true
	case "U16", "I16", "F16", "BF16":
		return 2, true
	case "U32", "I32", "F32":
		return 4, true
	case "U64", "I64", "F64":
		return 8, true
	default:
		return 0, false
	}
}

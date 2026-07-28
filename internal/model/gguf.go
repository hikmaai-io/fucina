package model

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	ggufMagic            = 0x46554747
	maxGGUFHeaderBytes   = 256 << 20
	maxGGUFKVCount       = 1 << 20
	maxGGUFTensorCount   = 1 << 20
	maxGGUFArrayElements = 1 << 20
	maxGGUFStringBytes   = 64 << 20
	maxGGUFTensorDims    = 8
)

const (
	ggufUint8 uint32 = iota
	ggufInt8
	ggufUint16
	ggufInt16
	ggufUint32
	ggufInt32
	ggufFloat32
	ggufBool
	ggufString
	ggufArray
	ggufUint64
	ggufInt64
	ggufFloat64
)

type ggufDecoder struct {
	r         io.Reader
	remaining int64
}

func (d *ggufDecoder) readFull(dst []byte) error {
	if int64(len(dst)) > d.remaining {
		return fmt.Errorf("GGUF header exceeds %d bytes", int64(maxGGUFHeaderBytes))
	}
	if _, err := io.ReadFull(d.r, dst); err != nil {
		return fmt.Errorf("truncated GGUF header: %w", err)
	}
	d.remaining -= int64(len(dst))
	return nil
}

func (d *ggufDecoder) u8() (uint8, error) {
	var b [1]byte
	if err := d.readFull(b[:]); err != nil {
		return 0, err
	}
	return b[0], nil
}

func (d *ggufDecoder) u16() (uint16, error) {
	var b [2]byte
	if err := d.readFull(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(b[:]), nil
}

func (d *ggufDecoder) u32() (uint32, error) {
	var b [4]byte
	if err := d.readFull(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func (d *ggufDecoder) u64() (uint64, error) {
	var b [8]byte
	if err := d.readFull(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func (d *ggufDecoder) str() (string, error) {
	n, err := d.u64()
	if err != nil {
		return "", err
	}
	if n > maxGGUFStringBytes || n > uint64(d.remaining) {
		return "", fmt.Errorf("invalid GGUF string length %d", n)
	}
	buf := make([]byte, int(n))
	if err := d.readFull(buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func (d *ggufDecoder) scalar(typ uint32) (any, error) {
	switch typ {
	case ggufUint8:
		return d.u8()
	case ggufInt8:
		n, err := d.u8()
		return int8(n), err
	case ggufUint16:
		return d.u16()
	case ggufInt16:
		n, err := d.u16()
		return int16(n), err
	case ggufUint32:
		return d.u32()
	case ggufInt32:
		n, err := d.u32()
		return int32(n), err
	case ggufFloat32:
		n, err := d.u32()
		return math.Float32frombits(n), err
	case ggufBool:
		n, err := d.u8()
		if err != nil {
			return nil, err
		}
		if n > 1 {
			return nil, fmt.Errorf("invalid GGUF bool %d", n)
		}
		return n == 1, nil
	case ggufString:
		return d.str()
	case ggufUint64:
		return d.u64()
	case ggufInt64:
		n, err := d.u64()
		return int64(n), err
	case ggufFloat64:
		n, err := d.u64()
		return math.Float64frombits(n), err
	default:
		return nil, fmt.Errorf("unsupported GGUF value type %d", typ)
	}
}

func (d *ggufDecoder) value(typ uint32) (any, error) {
	if typ != ggufArray {
		return d.scalar(typ)
	}
	elem, err := d.u32()
	if err != nil {
		return nil, err
	}
	if elem == ggufArray {
		return nil, fmt.Errorf("nested GGUF arrays are invalid")
	}
	n, err := d.u64()
	if err != nil {
		return nil, err
	}
	if n > maxGGUFArrayElements {
		return nil, fmt.Errorf("GGUF array has %d elements (limit %d)", n, maxGGUFArrayElements)
	}
	out := make([]any, int(n))
	for i := range out {
		out[i], err = d.scalar(elem)
		if err != nil {
			return nil, fmt.Errorf("GGUF array element %d: %w", i, err)
		}
	}
	return out, nil
}

// readGGUFMetadata reads only the bounded GGUF metadata and tensor-descriptor table.
// Tensor payload bytes are never read or allocated.
func readGGUFMetadata(path string) (GGUFMetadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return GGUFMetadata{}, fmt.Errorf("open GGUF: %w", err)
	}
	defer f.Close()

	d := &ggufDecoder{r: f, remaining: maxGGUFHeaderBytes}
	magic, err := d.u32()
	if err != nil {
		return GGUFMetadata{}, err
	}
	if magic != ggufMagic {
		return GGUFMetadata{}, fmt.Errorf("bad GGUF magic 0x%08x", magic)
	}
	version, err := d.u32()
	if err != nil {
		return GGUFMetadata{}, err
	}
	if version != 2 && version != 3 {
		return GGUFMetadata{}, fmt.Errorf("unsupported GGUF version %d (want 2 or 3)", version)
	}
	tensorCount, err := d.u64()
	if err != nil {
		return GGUFMetadata{}, err
	}
	kvCount, err := d.u64()
	if err != nil {
		return GGUFMetadata{}, err
	}
	if tensorCount > maxGGUFTensorCount {
		return GGUFMetadata{}, fmt.Errorf("GGUF tensor count %d exceeds limit %d", tensorCount, maxGGUFTensorCount)
	}
	if kvCount > maxGGUFKVCount {
		return GGUFMetadata{}, fmt.Errorf("GGUF metadata count %d exceeds limit %d", kvCount, maxGGUFKVCount)
	}

	out := GGUFMetadata{
		Values:  make(map[string]any, int(kvCount)),
		Tensors: make(map[string]TensorInfo, int(tensorCount)),
	}
	for i := uint64(0); i < kvCount; i++ {
		key, err := d.str()
		if err != nil {
			return GGUFMetadata{}, fmt.Errorf("GGUF metadata key %d: %w", i, err)
		}
		if _, exists := out.Values[key]; exists {
			return GGUFMetadata{}, fmt.Errorf("duplicate GGUF metadata key %q", key)
		}
		typ, err := d.u32()
		if err != nil {
			return GGUFMetadata{}, fmt.Errorf("GGUF metadata %q type: %w", key, err)
		}
		value, err := d.value(typ)
		if err != nil {
			return GGUFMetadata{}, fmt.Errorf("GGUF metadata %q: %w", key, err)
		}
		out.Values[key] = value
	}

	for i := uint64(0); i < tensorCount; i++ {
		name, err := d.str()
		if err != nil {
			return GGUFMetadata{}, fmt.Errorf("GGUF tensor %d name: %w", i, err)
		}
		if _, exists := out.Tensors[name]; exists {
			return GGUFMetadata{}, fmt.Errorf("duplicate GGUF tensor %q", name)
		}
		nDims, err := d.u32()
		if err != nil {
			return GGUFMetadata{}, fmt.Errorf("GGUF tensor %q dimensions: %w", name, err)
		}
		if nDims == 0 || nDims > maxGGUFTensorDims {
			return GGUFMetadata{}, fmt.Errorf("GGUF tensor %q has invalid rank %d", name, nDims)
		}
		shape := make([]int64, int(nDims))
		for j := range shape {
			dim, err := d.u64()
			if err != nil {
				return GGUFMetadata{}, fmt.Errorf("GGUF tensor %q dimension %d: %w", name, j, err)
			}
			if dim == 0 || dim > math.MaxInt64 {
				return GGUFMetadata{}, fmt.Errorf("GGUF tensor %q has invalid dimension %d", name, dim)
			}
			shape[j] = int64(dim)
		}
		ggmlType, err := d.u32()
		if err != nil {
			return GGUFMetadata{}, fmt.Errorf("GGUF tensor %q type: %w", name, err)
		}
		if _, err := d.u64(); err != nil { // payload-relative offset
			return GGUFMetadata{}, fmt.Errorf("GGUF tensor %q offset: %w", name, err)
		}
		out.Tensors[name] = TensorInfo{Shape: shape, Encoding: ggmlEncoding(ggmlType)}
	}
	return out, nil
}

func ggmlEncoding(typ uint32) string {
	if name, ok := map[uint32]string{
		0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
		8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
		13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
		18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
		23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
		29: "IQ1_M", 30: "BF16",
	}[typ]; ok {
		return name
	}
	return fmt.Sprintf("GGML_TYPE_%d", typ)
}

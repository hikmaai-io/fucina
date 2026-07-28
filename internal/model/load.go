package model

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// LoadDescriptor dispatches a model path to the source-format builder. It performs
// host-only metadata/tensor preflight and must run before CUDA engine allocation.
func LoadDescriptor(path string) (*ModelDescriptor, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("model manifest: %w", err)
	}
	if fi.IsDir() {
		return FromConfigJSON(path)
	}

	lower := strings.ToLower(path)
	switch {
	case strings.HasSuffix(lower, ".gguf"):
		metadata, err := readGGUFMetadata(path)
		if err != nil {
			return nil, fmt.Errorf("model manifest: %w", err)
		}
		return FromGGUF(metadata)
	case strings.HasSuffix(lower, ".safetensors"), strings.HasSuffix(lower, ".safetensors.index.json"):
		return FromConfigJSON(filepath.Dir(path))
	case filepath.Base(lower) == "config.json":
		return FromConfigJSON(path)
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("model manifest: %w", err)
	}
	defer f.Close()
	var magic [4]byte
	if _, err := f.Read(magic[:]); err == nil && binary.LittleEndian.Uint32(magic[:]) == ggufMagic {
		metadata, err := readGGUFMetadata(path)
		if err != nil {
			return nil, fmt.Errorf("model manifest: %w", err)
		}
		return FromGGUF(metadata)
	}
	return nil, fmt.Errorf("model manifest: unsupported checkpoint path %q", path)
}

// ABOUTME: Versioned on-disk session snapshot format (magic FUCINASESS1): JSON
// ABOUTME: metadata + token history + engine KV/state bytes, FNV-1a per section.
//
// A session file persists one conversation's engine state across process
// restarts so a later run can resume generation without re-prefilling the
// saved prefix. The state bytes are exactly what the engine's snapshot entry
// point produced (KVSave for the flat single-sequence engine, SeqStateSave for
// the Qwen3.5 hybrid per-slot path — the latter includes the GatedDeltaNet
// recurrent state and conv rings, without which a hybrid session cannot
// resume) and are treated as opaque here.
//
// Layout (all integers little-endian):
//
//	magic   12 B  "FUCINASESS1\0"
//	version u32
//	metaLen u32   | metaLen B JSON (Meta)      | u64 FNV-1a64(meta)
//	nTokens u32   | nTokens×4 B int32 history  | u64 FNV-1a64(tokens)
//	stateLen u64  | stateLen B engine state    | u64 FNV-1a64(state)
//	EOF (trailing bytes are an error)
//
// Load never trusts a length field from disk (the safetensors-loader
// standard): every section length is bounds-checked against a hard cap before
// allocation, checksums are verified, and the metadata token count must match
// the token section.
package session

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"os"
	"path/filepath"
	"time"
)

const (
	// Magic identifies a fucina session file; the trailing NUL pads it to 12
	// bytes so every integer after it is 4-byte aligned.
	Magic = "FUCINASESS1"
	// Version is the current format version; Load rejects any other.
	Version = 1

	maxMetaBytes  = 1 << 20 // JSON header cap
	maxTokens     = 1 << 26 // 67M tokens ≫ any context
	absStateBytes = 1 << 46 // absolute state cap (64 TiB); callers pass a real one
)

// Engine kinds recorded in the metadata. The two snapshot ABIs are not
// interchangeable, so a session saved by one never restores through the other.
const (
	KindFlatKV  = "flat-kv"  // gemma4_engine_kv_save: attention KV only
	KindQ35Slot = "q35-slot" // gemma4_engine_q35_state_save: KV + GDN state + conv rings
)

// Identity pins a session to the model that produced it.
type Identity struct {
	// Path is the model path as given on the CLI. Informational: two paths to
	// the same checkpoint validate via ConfigHash, not string equality.
	Path string `json:"path"`
	// ConfigHash is FNV-1a64 of the model's config bytes (config.json for a
	// checkpoint directory, the header window for a GGUF file).
	ConfigHash uint64 `json:"config_hash"`
	// StateProbe is the engine's snapshot size in bytes for a fixed
	// ProbeTokens-token sequence. It is a pure function of the loaded model's
	// geometry (layers, heads, dims, dtypes), so a mismatch means the state
	// bytes cannot be laid out into this engine even if the config hash lies.
	StateProbe int64 `json:"state_probe"`
}

// ProbeTokens is the fixed token count both save and load use to compute
// Identity.StateProbe.
const ProbeTokens = 16

// Meta is the JSON metadata header of a session file.
type Meta struct {
	CreatedAt  time.Time `json:"created_at"`
	NTokens    int       `json:"n_tokens"`
	EngineKind string    `json:"engine_kind"`
	Model      Identity  `json:"model"`
	// Client is an opaque blob for the saving client's own bookkeeping — the
	// REPL stores its chat history here so /load can rebuild future prompts
	// that token-match the saved sequence. The format validates its size (via
	// the metadata cap) but never its contents.
	Client json.RawMessage `json:"client,omitempty"`
}

// Snapshot is a fully validated, loaded session.
type Snapshot struct {
	Meta   Meta
	Tokens []int32
	State  []byte
}

// Validate checks that a session saved under meta m can restore into an
// engine currently described by cur. Path differences alone are fine.
func (m Meta) Validate(cur Meta) error {
	if m.EngineKind != cur.EngineKind {
		return fmt.Errorf("session engine kind %q does not match the loaded engine %q", m.EngineKind, cur.EngineKind)
	}
	if m.Model.ConfigHash != cur.Model.ConfigHash {
		return fmt.Errorf("session was saved for a different model config (hash %016x, loaded model %016x; session path %q, loaded path %q)",
			m.Model.ConfigHash, cur.Model.ConfigHash, m.Model.Path, cur.Model.Path)
	}
	if m.Model.StateProbe != cur.Model.StateProbe {
		return fmt.Errorf("session state geometry mismatch (probe %d bytes vs engine %d): the loaded engine lays out snapshot state differently",
			m.Model.StateProbe, cur.Model.StateProbe)
	}
	return nil
}

func fnv64(b []byte) uint64 {
	h := fnv.New64a()
	h.Write(b)
	return h.Sum64()
}

// Write serializes one session. tokens and state must both be nonempty — an
// empty session is never worth a file, and load-side zero lengths are treated
// as corruption.
func Write(w io.Writer, meta Meta, tokens []int32, state []byte) error {
	if len(tokens) == 0 {
		return fmt.Errorf("session: refusing to write empty token history")
	}
	if len(state) == 0 {
		return fmt.Errorf("session: refusing to write empty engine state")
	}
	if len(tokens) > maxTokens {
		return fmt.Errorf("session: token history %d exceeds format cap %d", len(tokens), maxTokens)
	}
	if meta.NTokens != len(tokens) {
		return fmt.Errorf("session: meta.NTokens %d != len(tokens) %d", meta.NTokens, len(tokens))
	}
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return fmt.Errorf("session: marshal meta: %w", err)
	}
	if len(metaJSON) > maxMetaBytes {
		return fmt.Errorf("session: metadata %d B exceeds cap %d", len(metaJSON), maxMetaBytes)
	}

	bw := bufio.NewWriter(w)
	var magic [12]byte
	copy(magic[:], Magic)
	bw.Write(magic[:])
	le := func(v any) { binary.Write(bw, binary.LittleEndian, v) }
	le(uint32(Version))

	le(uint32(len(metaJSON)))
	bw.Write(metaJSON)
	le(fnv64(metaJSON))

	tokBytes := make([]byte, 4*len(tokens))
	for i, t := range tokens {
		binary.LittleEndian.PutUint32(tokBytes[4*i:], uint32(t))
	}
	le(uint32(len(tokens)))
	bw.Write(tokBytes)
	le(fnv64(tokBytes))

	le(uint64(len(state)))
	bw.Write(state)
	le(fnv64(state))

	return bw.Flush()
}

// Read parses and fully validates a session. maxState caps the state section
// (pass the engine's expected snapshot size, or a generous bound); it exists
// so a hostile length field cannot force a huge allocation.
func Read(r io.Reader, maxState int64) (*Snapshot, error) {
	br := bufio.NewReader(r)

	var magic [12]byte
	if _, err := io.ReadFull(br, magic[:]); err != nil {
		return nil, fmt.Errorf("session: short read on magic: %w", err)
	}
	want := [12]byte{}
	copy(want[:], Magic)
	if magic != want {
		return nil, fmt.Errorf("session: bad magic %q — not a fucina session file", magic[:11])
	}
	var version uint32
	if err := binary.Read(br, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("session: short read on version: %w", err)
	}
	if version != Version {
		return nil, fmt.Errorf("session: unsupported format version %d (this build reads %d)", version, Version)
	}

	// readSection reads a length-prefixed, checksummed section with the length
	// already validated by the caller.
	readSection := func(n int64, what string) ([]byte, error) {
		b := make([]byte, n)
		if _, err := io.ReadFull(br, b); err != nil {
			return nil, fmt.Errorf("session: short read on %s (%d B): %w", what, n, err)
		}
		var sum uint64
		if err := binary.Read(br, binary.LittleEndian, &sum); err != nil {
			return nil, fmt.Errorf("session: short read on %s checksum: %w", what, err)
		}
		if got := fnv64(b); got != sum {
			return nil, fmt.Errorf("session: %s checksum mismatch (file %016x, computed %016x) — file is corrupt", what, sum, got)
		}
		return b, nil
	}

	var metaLen uint32
	if err := binary.Read(br, binary.LittleEndian, &metaLen); err != nil {
		return nil, fmt.Errorf("session: short read on metadata length: %w", err)
	}
	if metaLen == 0 || metaLen > maxMetaBytes {
		return nil, fmt.Errorf("session: metadata length %d outside (0, %d]", metaLen, maxMetaBytes)
	}
	metaJSON, err := readSection(int64(metaLen), "metadata")
	if err != nil {
		return nil, err
	}
	var meta Meta
	if err := json.Unmarshal(metaJSON, &meta); err != nil {
		return nil, fmt.Errorf("session: metadata is not valid JSON: %w", err)
	}

	var nTokens uint32
	if err := binary.Read(br, binary.LittleEndian, &nTokens); err != nil {
		return nil, fmt.Errorf("session: short read on token count: %w", err)
	}
	if nTokens == 0 || nTokens > maxTokens {
		return nil, fmt.Errorf("session: token count %d outside (0, %d]", nTokens, maxTokens)
	}
	if int(nTokens) != meta.NTokens {
		return nil, fmt.Errorf("session: token section holds %d tokens but metadata says %d", nTokens, meta.NTokens)
	}
	tokBytes, err := readSection(int64(nTokens)*4, "token history")
	if err != nil {
		return nil, err
	}
	tokens := make([]int32, nTokens)
	for i := range tokens {
		tokens[i] = int32(binary.LittleEndian.Uint32(tokBytes[4*i:]))
	}

	if maxState > absStateBytes {
		maxState = absStateBytes
	}
	var stateLen uint64
	if err := binary.Read(br, binary.LittleEndian, &stateLen); err != nil {
		return nil, fmt.Errorf("session: short read on state length: %w", err)
	}
	if stateLen == 0 || maxState <= 0 || stateLen > uint64(maxState) {
		return nil, fmt.Errorf("session: state length %d outside (0, %d]", stateLen, maxState)
	}
	state, err := readSection(int64(stateLen), "engine state")
	if err != nil {
		return nil, err
	}

	if _, err := br.ReadByte(); err != io.EOF {
		return nil, fmt.Errorf("session: trailing bytes after state checksum — file is corrupt or not a single session")
	}

	return &Snapshot{Meta: meta, Tokens: tokens, State: state}, nil
}

// WriteFile writes a session atomically (temp file + rename in the target
// directory), so a crash mid-save never leaves a truncated session behind.
func WriteFile(path string, meta Meta, tokens []int32, state []byte) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".fucina-sess-*")
	if err != nil {
		return fmt.Errorf("session: create temp in %s: %w", dir, err)
	}
	defer func() {
		if tmp != nil {
			tmp.Close()
			os.Remove(tmp.Name())
		}
	}()
	if err := Write(tmp, meta, tokens, state); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("session: fsync %s: %w", tmp.Name(), err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("session: close %s: %w", tmp.Name(), err)
	}
	name := tmp.Name()
	tmp = nil // disarm the cleanup defer
	if err := os.Rename(name, path); err != nil {
		os.Remove(name)
		return fmt.Errorf("session: rename into place: %w", err)
	}
	return nil
}

// ReadFile loads and validates a session file. maxState as in Read.
func ReadFile(path string, maxState int64) (*Snapshot, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("session: %w", err)
	}
	defer f.Close()
	return Read(f, maxState)
}

// ggufHashWindow is how much of a single-file model (GGUF) is hashed for the
// identity: the header + KV metadata live at the front; 1 MiB covers them
// while staying instant on a multi-GB file.
const ggufHashWindow = 1 << 20

// HashModelConfig fingerprints the model configuration behind modelPath: for
// a checkpoint directory it hashes config.json in full; for a single file it
// hashes the leading ggufHashWindow bytes (where GGUF keeps its metadata).
func HashModelConfig(modelPath string) (uint64, error) {
	fi, err := os.Stat(modelPath)
	if err != nil {
		return 0, fmt.Errorf("session: stat model: %w", err)
	}
	if fi.IsDir() {
		b, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
		if err != nil {
			return 0, fmt.Errorf("session: read model config: %w", err)
		}
		return fnv64(b), nil
	}
	f, err := os.Open(modelPath)
	if err != nil {
		return 0, fmt.Errorf("session: open model: %w", err)
	}
	defer f.Close()
	h := fnv.New64a()
	if _, err := io.Copy(h, io.LimitReader(f, ggufHashWindow)); err != nil {
		return 0, fmt.Errorf("session: hash model header: %w", err)
	}
	return h.Sum64(), nil
}

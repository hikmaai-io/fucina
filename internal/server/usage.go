package server

// CacheSource identifies where physically skipped prompt tokens came from.
type CacheSource string

const (
	CacheSourceNone          CacheSource = ""
	CacheSourceLiveSeq       CacheSource = "live-seq"
	CacheSourceGPUPagedBlock CacheSource = "gpu-paged-block"
	CacheSourceHostSnapshot  CacheSource = "host-snapshot"
	CacheSourceDiskSession   CacheSource = "disk-session"
)

// PromptTokensDetails is the OpenAI-compatible prompt cache detail payload.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// PromptAccounting tracks logical prompt size and the physically skipped subset.
type PromptAccounting struct {
	PromptTokens int
	CachedTokens int
	Source       CacheSource
}

// NewPrefillTokens is the prompt suffix that actually ran through the model.
func (p PromptAccounting) NewPrefillTokens() int {
	if p.CachedTokens <= 0 {
		return p.PromptTokens
	}
	if p.CachedTokens >= p.PromptTokens {
		return 0
	}
	return p.PromptTokens - p.CachedTokens
}

// Normalized clamps accounting to the required invariants.
func (p PromptAccounting) Normalized() PromptAccounting {
	if p.PromptTokens < 0 {
		p.PromptTokens = 0
	}
	if p.CachedTokens < 0 {
		p.CachedTokens = 0
	}
	if p.CachedTokens > p.PromptTokens {
		p.CachedTokens = p.PromptTokens
	}
	if p.CachedTokens == 0 {
		p.Source = CacheSourceNone
	}
	return p
}

// MergePromptAccounting keeps the larger physically skipped prefix for one request.
func (p PromptAccounting) PromptTokensDetails() *PromptTokensDetails {
	p = p.Normalized()
	if p.CachedTokens == 0 {
		return nil
	}
	return &PromptTokensDetails{CachedTokens: p.CachedTokens}
}

func MergePromptAccounting(a, b PromptAccounting) PromptAccounting {
	a = a.Normalized()
	b = b.Normalized()
	if b.PromptTokens > a.PromptTokens {
		a.PromptTokens = b.PromptTokens
	}
	if b.CachedTokens > a.CachedTokens {
		return b
	}
	return a
}

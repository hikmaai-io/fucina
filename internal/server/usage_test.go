package server

import "testing"

func TestPromptAccountingNormalized(t *testing.T) {
	got := (PromptAccounting{PromptTokens: 8, CachedTokens: 99, Source: CacheSourceHostSnapshot}).Normalized()
	if got.CachedTokens != 8 {
		t.Fatalf("cached=%d want 8", got.CachedTokens)
	}
	if got.NewPrefillTokens() != 0 {
		t.Fatalf("new_prefill=%d want 0", got.NewPrefillTokens())
	}

	zero := (PromptAccounting{PromptTokens: 5, CachedTokens: -1, Source: CacheSourceLiveSeq}).Normalized()
	if zero.CachedTokens != 0 || zero.Source != CacheSourceNone {
		t.Fatalf("zero=%+v want cached=0 source empty", zero)
	}
	if zero.NewPrefillTokens() != 5 {
		t.Fatalf("new_prefill=%d want 5", zero.NewPrefillTokens())
	}
	if details := got.PromptTokensDetails(); details == nil || details.CachedTokens != 8 {
		t.Fatalf("details=%+v want cached=8", details)
	}
	if details := zero.PromptTokensDetails(); details != nil {
		t.Fatalf("zero details=%+v want nil", details)
	}
}

func TestMergePromptAccountingKeepsLargestPhysicalSkip(t *testing.T) {
	a := PromptAccounting{PromptTokens: 1024, CachedTokens: 256, Source: CacheSourceGPUPagedBlock}
	b := PromptAccounting{PromptTokens: 1024, CachedTokens: 768, Source: CacheSourceDiskSession}
	got := MergePromptAccounting(a, b)
	if got.CachedTokens != 768 || got.Source != CacheSourceDiskSession {
		t.Fatalf("merged=%+v", got)
	}
	if got.NewPrefillTokens() != 256 {
		t.Fatalf("new_prefill=%d want 256", got.NewPrefillTokens())
	}
}

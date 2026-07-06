package batch

import "testing"

func TestPromptLookupDraftFindsRepeat(t *testing.T) {
	// "a b c d e a b c" — the suffix [a b c] recurs; its earlier continuation is d.
	hist := []int32{10, 11, 12, 13, 14, 10, 11, 12}
	d := promptLookupDraft(hist, 4, 2, 4)
	if len(d) == 0 || d[0] != 13 {
		t.Fatalf("draft = %v, want it to start with 13 (the token after the earlier [10 11 12])", d)
	}
}

func TestPromptLookupDraftNoMatchOnNovelText(t *testing.T) {
	// Strictly increasing: no repeated >=2-gram, so nothing should be proposed.
	hist := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	if d := promptLookupDraft(hist, 4, 2, 4); len(d) != 0 {
		t.Fatalf("draft = %v, want empty on novel text", d)
	}
}

func TestPromptLookupDraftRespectsMaxD(t *testing.T) {
	// A long verbatim repeat: continuation is long, but maxD caps the draft length.
	base := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	hist := append(append([]int32{}, base...), base[:3]...) // ...1 2 3 again
	d := promptLookupDraft(hist, 2, 2, 6)
	if len(d) > 2 {
		t.Fatalf("draft len = %d, want <= maxD=2 (got %v)", len(d), d)
	}
}

func TestPromptLookupDraftZeroBudget(t *testing.T) {
	hist := []int32{10, 11, 12, 10, 11}
	if d := promptLookupDraft(hist, 0, 2, 4); d != nil {
		t.Fatalf("draft = %v, want nil at maxD=0", d)
	}
}

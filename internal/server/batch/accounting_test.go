package batch

import "testing"

func TestReuseInfoNormalized(t *testing.T) {
	got := (ReuseInfo{ReusedTokens: -3, Source: ReuseSourceHostSnapshot}).Normalized()
	if got.ReusedTokens != 0 || got.Source != ReuseSourceNone {
		t.Fatalf("normalized=%+v", got)
	}
}

func TestPagedOrdinaryAdoptionCountsOnlyCompleteBlocks(t *testing.T) {
	if got := AdoptedCachedTokens(300); got != 256 {
		t.Fatalf("paged 300 shared -> %d want 256", got)
	}
}

func TestAdoptedCachedTokens(t *testing.T) {
	for _, tc := range []struct {
		shared int
		want   int
	}{
		{shared: -1, want: 0},
		{shared: 0, want: 0},
		{shared: 1, want: 0},
		{shared: 255, want: 0},
		{shared: 256, want: 256},
		{shared: 511, want: 256},
		{shared: 512, want: 512},
		{shared: 767, want: 512},
	} {
		if got := AdoptedCachedTokens(tc.shared); got != tc.want {
			t.Fatalf("shared=%d got=%d want=%d", tc.shared, got, tc.want)
		}
	}
}

func TestResidentCachedPrefixRequiresExactStrictPrefixAndReportsFullBlocks(t *testing.T) {
	resident := make([]int32, 3021)
	for i := range resident {
		resident[i] = int32(i)
	}
	prompt := append(append([]int32(nil), resident...), 99)
	physical, reported := ResidentCachedPrefix(resident, prompt)
	if physical != 3021 || reported != 2816 {
		t.Fatalf("physical=%d reported=%d", physical, reported)
	}
	prompt[17]++
	if physical, reported := ResidentCachedPrefix(resident, prompt); physical != 0 || reported != 0 {
		t.Fatalf("mismatch reused physical=%d reported=%d", physical, reported)
	}
	if physical, reported := ResidentCachedPrefix(resident, resident); physical != 0 || reported != 0 {
		t.Fatalf("non-strict prefix reused physical=%d reported=%d", physical, reported)
	}
}

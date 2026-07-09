package main

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestJSpaceJSONUsesDecodedTokenWords(t *testing.T) {
	data, err := json.Marshal(jspaceTopToken{Token: " Paris", TokenID: 42, Prob: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	got := string(data)
	if !strings.Contains(got, `"token":" Paris"`) || !strings.Contains(got, `"token_id":42`) {
		t.Fatalf("human-readable token missing from %s", got)
	}
	if strings.Contains(got, `"text"`) || strings.Contains(got, `"id":`) {
		t.Fatalf("legacy number-first schema leaked into %s", got)
	}
}

func TestJSpaceRecordSeparatesResidualSourceFromSampledToken(t *testing.T) {
	data, err := json.Marshal(jspaceRecord{SourcePosition: 12, SourceToken: "ook",
		SourceTokenID: 7, SampledToken: "edly", SampledTokenID: 8})
	if err != nil {
		t.Fatal(err)
	}
	got := string(data)
	for _, field := range []string{`"source_token":"ook"`, `"source_token_id":7`,
		`"sampled_token":"edly"`, `"sampled_token_id":8`} {
		if !strings.Contains(got, field) {
			t.Fatalf("causal trace field %s missing from %s", field, got)
		}
	}
}

func TestSplitJSpaceCommandPreservesTokenWords(t *testing.T) {
	got, err := splitJSpaceCommand(`/jsteer " Paris" -0.15 8,16,24`)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"/jsteer", " Paris", "-0.15", "8,16,24"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("split=%q, want %q", got, want)
	}
	if _, err := splitJSpaceCommand(`/jsteer "broken`); err == nil {
		t.Error("unterminated quote accepted")
	}
}

func TestParseJSpaceLayers(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want []int
	}{
		{"", nil},
		{"all", nil},
		{"0", []int{0}},
		{"3, 7,31", []int{3, 7, 31}},
	} {
		got, err := parseJSpaceLayers(tc.in)
		if err != nil {
			t.Fatalf("parseJSpaceLayers(%q): %v", tc.in, err)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("parseJSpaceLayers(%q)=%v, want %v", tc.in, got, tc.want)
		}
	}
	if _, err := parseJSpaceLayers("2,nope"); err == nil {
		t.Error("invalid layer accepted")
	}
}
